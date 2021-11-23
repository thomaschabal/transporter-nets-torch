# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Ground-truth state Agent."""

import os
import time  # pylint: disable=g-import-not-at-top

import matplotlib.pyplot as plt
import numpy as np
from ravens_torch.models import mdn_utils
from ravens_torch.models.gt_state import MlpModel
from ravens_torch.tasks import cameras
from ravens_torch.utils import utils, MeanMetrics, to_device
from transforms3d import quaternions
import torch
import torch.nn as nn
import torch.optim as optim


class GtStateAgent:
    """Agent which uses ground-truth state information -- useful as a baseline."""

    def __init__(self, name, task, root_dir, verbose=False):
        # self.fig, self.ax = plt.subplots(1,1)
        self.name = name
        self.task = task

        if self.task in ['aligning', 'palletizing', 'packing']:
            self.use_box_dimensions = True
        else:
            self.use_box_dimensions = False

        if self.task in ['sorting']:
            self.use_colors = True
        else:
            self.use_colors = False

        self.total_steps = 0
        self.camera_config = cameras.RealSenseD415.CONFIG

        # A place to save pre-trained models.
        self.models_dir = os.path.join(root_dir, 'checkpoints', self.name)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        # Set up model.
        self.model = None
        # boundaries = [1000, 2000, 5000, 10000]
        # values = [1e-2, 1e-2, 1e-2, 1e-3, 1e-5]
        # learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #   boundaries, values)
        # self.optim = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
        self.optim = None
        self.verbose = verbose

        self.metric = MeanMetrics()
        self.val_metric = MeanMetrics()
        self.theta_scale = 10.0
        self.batch_size = 128
        self.use_mdn = True

        self.criterion = mdn_utils.mdn_loss if self.use_mdn else nn.MSELoss()

        # self.vis = utils.create_visualizer()

        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
        self.pixel_size = 0.003125
        self.six_dof = False

    def extract_x_y_theta(self,
                          object_info,
                          t_worldaug_world=None,
                          preserve_theta=False):
        """Extract in-plane theta."""
        object_position = object_info[0]
        object_quat_xyzw = object_info[1]

        if t_worldaug_world is not None:
            object_quat_wxyz = (object_quat_xyzw[3], object_quat_xyzw[0],
                                object_quat_xyzw[1], object_quat_xyzw[2])
            t_world_object = quaternions.quat2mat(object_quat_wxyz)
            temp_t_world_object = np.eye(4)
            temp_t_world_object[:3, :3] = t_world_object
            t_world_object = np.copy(temp_t_world_object)
            import pudb
            pudb.set_trace()
            t_world_object[0:3, 3] = np.array(object_position)
            t_worldaug_object = t_worldaug_world @ t_world_object

            object_quat_wxyz = quaternions.mat2quat(t_worldaug_object)
            if not preserve_theta:
                object_quat_xyzw = (object_quat_wxyz[1], object_quat_wxyz[2],
                                    object_quat_wxyz[3], object_quat_wxyz[0])
            object_position = t_worldaug_object[0:3, 3]

        object_xy = object_position[0:2]
        object_theta = -np.float32(
            utils.quatXYZW_to_eulerXYZ(object_quat_xyzw)
            [2]) / self.theta_scale
        return np.hstack(
            (object_xy,
             object_theta)).astype(np.float32), object_position, object_quat_xyzw

    def extract_box_dimensions(self, info):
        return np.array(info[2])

    def extract_color(self, info):
        return np.array(info[-1])

    def info_to_gt_obs(self, info, t_worldaug_world=None):
        """Calculate ground-truth observation vector for environment info."""
        object_keys = sorted(info.keys())
        observation_vector = []
        for object_key in object_keys:
            object_x_y_theta, _, _ = self.extract_x_y_theta(
                info[object_key], t_worldaug_world)
            observation_vector.append(object_x_y_theta)
            if self.use_box_dimensions:
                observation_vector.append(
                    self.extract_box_dimensions(info[object_key]))
            if self.use_colors:
                observation_vector.append(self.extract_color(info[object_key]))
        observation_vector = np.array(observation_vector).reshape(-1).astype(
            np.float32)

        # pad with zeros
        if self.max_obs_vector_length != 0:
            observation_vector = np.pad(
                observation_vector,
                [0, self.max_obs_vector_length - len(observation_vector)])

        return observation_vector

    def act_to_gt_act(self, act, t_worldaug_world=None, transform_params=None):
        del transform_params

        # dont update theta due to suction invariance to theta
        pick_se2, _, _ = self.extract_x_y_theta(
            act['params']['pose0'], t_worldaug_world, preserve_theta=True)
        place_se2, _, _ = self.extract_x_y_theta(
            act['params']['pose1'], t_worldaug_world, preserve_theta=True)
        return np.hstack((pick_se2, place_se2)).astype(np.float32)

    def set_max_obs_vector_length(self, dataset):
        num_samples = 2000  # just to find the largest environment dimensionality
        self.max_obs_vector_length = 0
        max_obs_vector_length = 0
        for _ in range(num_samples):
            (_, _, _, info), _ = dataset.sample()
            obs_vector_length = self.info_to_gt_obs(info).shape[0]
            if obs_vector_length > max_obs_vector_length:
                max_obs_vector_length = obs_vector_length
        self.max_obs_vector_length = max_obs_vector_length

    def init_model(self, dataset):
        """Initialize self.model, including normalization parameters."""
        self.set_max_obs_vector_length(dataset)

        (_, _, _, info), _ = dataset.sample()
        obs_vector = self.info_to_gt_obs(info)

        obs_dim = obs_vector.shape[0]
        act_dim = 6
        if self.six_dof:
            act_dim = 9
        self.model = MlpModel(
            obs_dim, act_dim, 'relu', self.use_mdn, dropout=0.1)
        self.device = to_device([self.model], "GT State", verbose=self.verbose)

        sampled_gt_obs = []

        num_samples = 1000
        for _ in range(num_samples):
            (_, _, _, info), _ = dataset.sample()
            t_worldaug_world, _ = self.get_augmentation_transform()
            sampled_gt_obs.append(self.info_to_gt_obs(info, t_worldaug_world))

        sampled_gt_obs = np.array(sampled_gt_obs)

        obs_train_parameters = dict()
        obs_train_parameters['mean'] = sampled_gt_obs.mean(axis=(0)).astype(
            np.float32)
        obs_train_parameters['std'] = sampled_gt_obs.std(axis=(0)).astype(
            np.float32)
        self.model.set_normalization_parameters(obs_train_parameters)

        self.optim = optim.Adam(self.model.parameters(), lr=2e-4)

    def get_augmentation_transform(self):
        heightmap = np.zeros((320, 160))
        theta, trans, pivot = utils.get_random_image_transform_params(
            heightmap.shape)
        transform_params = theta, trans, pivot
        t_world_center, t_world_centeraug = utils.get_se3_from_image_transform(
            *transform_params, heightmap, self.bounds, self.pixel_size)
        t_worldaug_world = t_world_centeraug @ np.linalg.inv(t_world_center)
        return t_worldaug_world, transform_params

    def get_data_batch(self, dataset):
        """Pre-process info and obs-act, and make batch."""
        batch_obs = []
        batch_act = []
        for _ in range(self.batch_size):
            (obs, act, _, info), _ = dataset.sample()
            t_worldaug_world, transform_params = self.get_augmentation_transform()

            batch_obs.append(self.info_to_gt_obs(info, t_worldaug_world))
            batch_act.append(
                self.act_to_gt_act(
                    act, t_worldaug_world,
                    transform_params))  # this samples pick points from surface
            # on insertion task only, this can be used to imagine as if the picks were
            # deterministic.
            # batch_act.append(self.info_to_gt_obs(info))

        batch_obs = np.array(batch_obs)
        batch_act = np.array(batch_act)
        return batch_obs, batch_act, obs, act, info

    def train(self, dataset, writer):
        """Train on dataset for a specific number of iterations."""
        if self.model is None:
            self.init_model(dataset)

        self.transport.train_mode()
        self.metric.reset()
        self.optim.zero_grad()

        start = time.time()
        batch_obs, batch_act, _, _, _ = self.get_data_batch(dataset)

        # Forward through model, compute training loss, update weights.
        prediction = self.model(batch_obs)
        loss = self.criterion(batch_act, prediction)

        loss.backward()
        self.optim.step()
        self.metric(loss)
        writer.add_scalar('train/gt_state_loss',
                          self.metric.result(), step=self.total_steps + i)

        loss = np.float32(loss)
        print(f'Train Iter: {self.total_steps + i} Loss: {loss:.4f} Iter time:',
              time.time() - start)
        # utils.meshcat_visualize(self.vis, obs, act, info)

        self.total_steps += 1

    def validate(self, dataset, writer):
        """Train on dataset for a specific number of iterations."""

        if self.model is None:
            self.init_model(dataset)

        # Compute valid loss
        print('Validating!')
        self.model.eval()
        self.val_metric.reset()

        batch_obs, batch_act, _, _, _ = self.get_data_batch(dataset)
        with torch.no_grad():
            prediction = self.model(batch_obs)
            loss = self.criterion(batch_act, prediction)

        self.val_metric(loss)
        writer.add_scalar(
            'validation/gt_state_loss', self.val_metric.result(), step=self.total_steps + i)

        self.total_steps += 1

    def plot_act_mdn(self, y, mdn_predictions):
        """Plot actions.

        Args:
          y: true "y", shape (batch_size, d_out)
          mdn_predictions: tuple of:
            - pi: (batch_size, num_gaussians)
            - mu: (batch_size, num_gaussians * d_out)
            - var: (batch_size, num_gaussians)
        """

        pi, mu, _ = mdn_predictions

        self.ax.cla()
        self.ax.scatter(y[:, 0], y[:, 1])
        mu = torch.reshape(mu, (-1, y.shape[-1]))
        pi = torch.reshape(pi, (-1,))

        pi = torch.clip(pi, 0.01, 1.0)

        rgba_colors = np.zeros((len(pi), 4))
        # for red the first column needs to be one
        rgba_colors[:, 0] = 1.0
        # the fourth column needs to be your alphas
        rgba_colors[:, 3] = pi

        self.ax.scatter(mu[:, 0], mu[:, 1], color=rgba_colors)

        plt.draw()
        plt.pause(0.001)

    def act(self, obs, info):
        """Run inference and return best action."""
        del obs

        act = {'camera_config': self.camera_config, 'primitive': None}

        # Get observations and run predictions.
        gt_obs = self.info_to_gt_obs(info)

        # just for visualization
        gt_act_center = self.info_to_gt_obs(
            info)  # pylint: disable=unused-variable

        prediction = self.model(gt_obs[None, Ellipsis])

        if self.use_mdn:
            mdn_prediction = prediction
            pi, mu, var = mdn_prediction
            # prediction = mdn_utils.pick_max_mean(pi, mu, var)
            prediction = mdn_utils.sample_from_pdf(pi, mu, var)
            prediction = prediction[:, 0, :]

        prediction = prediction[0]  # unbatch

        # self.plot_act_mdn(gt_act_center[None, ...], mdn_prediction)

        # just go exactly to objects, from observations
        # p0_position = np.hstack((gt_obs[3:5], 0.02))
        # p0_rotation = utils.eulerXYZ_to_quatXYZW(
        #     (0, 0, -gt_obs[5] * self.theta_scale))
        # p1_position = np.hstack((gt_obs[0:2], 0.02))
        # p1_rotation = utils.eulerXYZ_to_quatXYZW(
        #     (0, 0, -gt_obs[2] * self.theta_scale))

        # just go exactly to objects, predicted
        p0_position = np.hstack((prediction[0:2], 0.02))
        p0_rotation = utils.eulerXYZ_to_quatXYZW(
            (0, 0, -prediction[2] * self.theta_scale))
        p1_position = np.hstack((prediction[3:5], 0.02))
        p1_rotation = utils.eulerXYZ_to_quatXYZW(
            (0, 0, -prediction[5] * self.theta_scale))

        # Select task-specific motion primitive.
        act['primitive'] = 'pick_place'
        if self.task == 'sweeping':
            act['primitive'] = 'sweep'
        elif self.task == 'pushing':
            act['primitive'] = 'push'

        params = {
            'pose0': (np.asarray(p0_position), np.asarray(p0_rotation)),
            'pose1': (np.asarray(p1_position), np.asarray(p1_rotation))
        }
        act['params'] = params
        return act

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------

    def load(self, num_iter, verbose=False):
        """Load something."""

        # Do something here.
        # self.model.load(os.path.join(self.models_dir, model_fname))
        # Update total training iterations of agent.
        self.total_steps = num_iter

    def save(self, verbose=False):
        """Save models."""
        # Do something here.
        # self.model.save(os.path.join(self.models_dir, model_fname))
        pass


class GtState6DAgent(GtStateAgent):
    """Agent which uses ground-truth state information -- useful as a baseline."""

    def __init__(self, name, task):
        super(GtState6DAgent, self).__init__(name, task)
        self.six_dof = True

    def act(self, obs, gt_act, info):
        """Run inference and return best action."""
        del gt_act

        assert False, 'this needs to have the ordering switched for act inference -- is now xytheta, rpz'  # pylint: disable=line-too-long
        act = {'camera_config': self.camera_config, 'primitive': None}

        # Get observations and run predictions.
        gt_obs = self.info_to_gt_obs(info)
        prediction = self.model(gt_obs[None, Ellipsis])

        if self.use_mdn:
            mdn_prediction = prediction
            pi, mu, var = mdn_prediction
            # prediction = mdn_utils.pick_max_mean(pi, mu, var)
            prediction = mdn_utils.sample_from_pdf(pi, mu, var)
            prediction = prediction[:, 0, :]

        prediction = prediction[0]  # unbatch

        p0_position = np.hstack((prediction[0:2], 0.02))
        p0_rotation = utils.eulerXYZ_to_quatXYZW(
            (0, 0, -prediction[2] * self.theta_scale))

        p1_position = prediction[3:6]
        p1_rotation = utils.eulerXYZ_to_quatXYZW(
            (prediction[6] * self.theta_scale, prediction[7] * self.theta_scale,
             -prediction[8] * self.theta_scale))

        # Select task-specific motion primitive.
        act['primitive'] = 'pick_place_6dof'

        params = {
            'pose0': (p0_position, p0_rotation),
            'pose1': (p1_position, p1_rotation)
        }
        act['params'] = params
        return act

    def info_to_gt_obs(self, info, t_worldaug_world=None):
        object_keys = sorted(info.keys())

        observation_vector = []
        for object_key in object_keys:
            object_xyzrpy = self.get_six_dof_object(info[object_key],
                                                    t_worldaug_world)
            observation_vector.append(object_xyzrpy)
            if self.use_box_dimensions:
                observation_vector.append(
                    self.extract_box_dimensions(info[object_key]))
        observation_vector = np.array(observation_vector).reshape(-1).astype(
            np.float32)

        # pad with zeros
        if self.max_obs_vector_length != 0:
            observation_vector = np.pad(
                observation_vector,
                [0, self.max_obs_vector_length - len(observation_vector)])

        return observation_vector

    def get_six_dof_object(self, object_info, t_worldaug_world=None):
        """Calculate the pose of 6DOF object."""
        object_position = object_info[0]
        object_quat_xyzw = object_info[1]

        if t_worldaug_world is not None:
            object_quat_wxyz = (object_quat_xyzw[3], object_quat_xyzw[0],
                                object_quat_xyzw[1], object_quat_xyzw[2])
            t_world_object = quaternions.quat2mat(object_quat_wxyz)
            t_world_object[0:3, 3] = np.array(object_position)
            t_worldaug_object = t_worldaug_world @ t_world_object

            object_quat_wxyz = quaternions.mat2quat(
                t_worldaug_object)
            object_quat_xyzw = (object_quat_wxyz[1], object_quat_wxyz[2],
                                object_quat_wxyz[3], object_quat_wxyz[0])
            object_position = t_worldaug_object[0:3, 3]

        euler = utils.quatXYZW_to_eulerXYZ(object_quat_xyzw)
        roll = euler[0]
        pitch = euler[1]
        theta = -euler[2]

        return np.asarray([
            object_position[0], object_position[1], object_position[2], roll, pitch,
            theta
        ])

    def act_to_gt_act(self, act, t_worldaug_world=None, transform_params=None):
        # dont update theta due to suction invariance to theta
        pick_se2, _, _ = self.extract_x_y_theta(
            act['params']['pose0'], t_worldaug_world, preserve_theta=True)
        heightmap = np.zeros((320, 160))
        place_se3 = self.get_six_dof_act(transform_params, heightmap,
                                         act['params']['pose0'],
                                         act['params']['pose1'])
        return np.hstack((pick_se2, place_se3)).astype(np.float32)

    def get_six_dof_act(self, transform_params, heightmap, pose0, pose1):
        """Adjust SE(3) poses via the in-plane SE(2) augmentation transform."""
        p1_position, p1_rotation = pose1[0], pose1[1]
        p0_position, p0_rotation = pose0[0], pose0[1]

        if transform_params is not None:
            t_world_center, t_world_centernew = utils.get_se3_from_image_transform(
                transform_params[0], transform_params[1], transform_params[2],
                heightmap, self.bounds, self.pixel_size)
            t_worldnew_world = t_world_centernew @ np.linalg.inv(
                t_world_center)
        else:
            t_worldnew_world = np.eye(4)

        p1_quat_wxyz = (p1_rotation[3], p1_rotation[0], p1_rotation[1],
                        p1_rotation[2])
        t_world_p1 = quaternions.quat2mat(p1_quat_wxyz)
        t_world_p1[0:3, 3] = np.array(p1_position)

        t_worldnew_p1 = t_worldnew_world @ t_world_p1

        p0_quat_wxyz = (p0_rotation[3], p0_rotation[0], p0_rotation[1],
                        p0_rotation[2])
        t_world_p0 = quaternions.quat2mat(p0_quat_wxyz)
        t_world_p0[0:3, 3] = np.array(p0_position)
        t_worldnew_p0 = t_worldnew_world @ t_world_p0

        t_worldnew_p0theta0 = t_worldnew_p0 * 1.0
        t_worldnew_p0theta0[0:3, 0:3] = np.eye(3)

        # PLACE FRAME, adjusted for this 0 rotation on pick
        t_p0_p0theta0 = np.linalg.inv(t_worldnew_p0) @ t_worldnew_p0theta0
        t_worldnew_p1theta0 = t_worldnew_p1 @ t_p0_p0theta0

        # convert the above rotation to euler
        quatwxyz_worldnew_p1theta0 = quaternions.mat2quat(
            t_worldnew_p1theta0)
        q = quatwxyz_worldnew_p1theta0
        quatxyzw_worldnew_p1theta0 = (q[1], q[2], q[3], q[0])
        p1_rotation = quatxyzw_worldnew_p1theta0
        p1_euler = utils.quatXYZW_to_eulerXYZ(p1_rotation)

        roll_scaled = p1_euler[0] / self.theta_scale
        pitch_scaled = p1_euler[1] / self.theta_scale
        p1_theta_scaled = -p1_euler[2] / self.theta_scale

        x = p1_position[0]
        y = p1_position[1]
        z = p1_position[2]

        return np.array([x, y, p1_theta_scaled, roll_scaled, pitch_scaled, z])
