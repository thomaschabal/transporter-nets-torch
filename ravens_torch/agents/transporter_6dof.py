# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Transporter Agent (6DoF Hybrid with Regression)."""

import numpy as np
from ravens_torch import models
from ravens_torch.agents.transporter import TransporterAgent
from ravens_torch.utils import utils, MeanMetrics
from transforms3d import quaternions


class Transporter6dAgent(TransporterAgent):
    """6D Transporter variant."""

    def __init__(self, name, task, root_dir, verbose=False):
        super().__init__(name, task, root_dir)

        self.attention_model = models.Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            verbose=verbose)
        self.transport_model = models.Transport(
            in_channels=self.in_shape[2],
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            # six_dof=False,
            verbose=verbose)

        self.rpz_model = models.Transport(
            in_channels=self.in_shape[2],
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            # six_dof=True,
            verbose=verbose,
            name="RPZ Transport")

        # self.transport_model.set_bounds_pix_size(
        #     self.bounds, self.pix_size)
        # self.rpz_model.set_bounds_pix_size(self.bounds, self.pix_size)

        self.six_dof = True

        self.p0_pixel_error = MeanMetrics()
        self.p1_pixel_error = MeanMetrics()
        self.p0_theta_error = MeanMetrics()
        self.p1_theta_error = MeanMetrics()
        self.metrics = [
            self.p0_pixel_error, self.p1_pixel_error, self.p0_theta_error,
            self.p1_theta_error
        ]

    def get_six_dof(self,
                    transform_params,
                    heightmap,
                    pose0,
                    pose1,
                    augment=True):
        """Adjust SE(3) poses via the in-plane SE(2) augmentation transform."""

        p1_position, p1_rotation = pose1[0], pose1[1]
        p0_position, p0_rotation = pose0[0], pose0[1]

        if augment:
            t_world_center, t_world_centernew = utils.get_se3_from_image_transform(
                *transform_params, heightmap, self.bounds, self.pix_size)

            t_worldnew_world = t_world_centernew @ np.linalg.inv(
                t_world_center)
        else:
            t_worldnew_world = np.eye(4)

        p1_quat_wxyz = (p1_rotation[3], p1_rotation[0], p1_rotation[1],
                        p1_rotation[2])
        t_world_p1 = quaternions.quat2mat(p1_quat_wxyz)

        tmp_t_world_p1 = np.eye(4)
        tmp_t_world_p1[:3, :3] = t_world_p1
        t_world_p1 = np.copy(tmp_t_world_p1)
        t_world_p1[0:3, 3] = np.array(p1_position)

        t_worldnew_p1 = t_worldnew_world @ t_world_p1

        p0_quat_wxyz = (p0_rotation[3], p0_rotation[0], p0_rotation[1],
                        p0_rotation[2])
        t_world_p0 = quaternions.quat2mat(p0_quat_wxyz)
        tmp_t_world_p0 = np.eye(4)
        tmp_t_world_p0[:3, :3] = t_world_p0
        t_world_p0 = np.copy(tmp_t_world_p0)
        t_world_p0[0:3, 3] = np.array(p0_position)
        t_worldnew_p0 = t_worldnew_world @ t_world_p0

        # PICK FRAME, using 0 rotation due to suction rotational symmetry
        t_worldnew_p0theta0 = t_worldnew_p0 * 1.0
        t_worldnew_p0theta0[0:3, 0:3] = np.eye(3)

        # PLACE FRAME, adjusted for this 0 rotation on pick
        t_p0_p0theta0 = np.linalg.inv(t_worldnew_p0) @ t_worldnew_p0theta0
        t_worldnew_p1theta0 = t_worldnew_p1 @ t_p0_p0theta0

        # convert the above rotation to euler
        quatwxyz_worldnew_p1theta0 = quaternions.mat2quat(
            t_worldnew_p1theta0[:3, :3])
        q = quatwxyz_worldnew_p1theta0
        quatxyzw_worldnew_p1theta0 = (q[1], q[2], q[3], q[0])
        p1_rotation = quatxyzw_worldnew_p1theta0
        p1_euler = utils.quatXYZW_to_eulerXYZ(p1_rotation)
        roll = p1_euler[0]
        pitch = p1_euler[1]
        p1_theta = -p1_euler[2]

        p0_theta = 0
        z = p1_position[2]
        return p0_theta, p1_theta, z, roll, pitch

    def get_sample(self, dataset, augment=True):
        (obs, act, _, _), _ = dataset.sample()
        img = self.get_image(obs)

        # Get training labels from data sample.
        p0_xyz, p0_xyzw = act['pose0']
        p1_xyz, p1_xyzw = act['pose1']
        p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
        p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
        p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
        p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
        p1_theta = p1_theta - p0_theta
        p0_theta = 0

        if augment:
            img, _, (p0, p1), transforms = utils.perturb(img, [p0, p1])
            p0_theta, p1_theta, z, roll, pitch = self.get_six_dof(
                transforms, img[:, :, 3], (p0_xyz, p0_xyzw), (p1_xyz, p1_xyzw))

        return img, p0, p0_theta, p1, p1_theta, z, roll, pitch

    def train(self, dataset, writer):
        """Train on dataset for a specific number of iterations.

        Args:
          dataset: a ravens_torch.Dataset.
          writer: a TensorboardX SummaryWriter.
        """
        self.attention_model.train_mode()
        self.transport_model.train_mode()
        self.rpz_model.train_mode()

        input_image, p0, p0_theta, p1, p1_theta, z, roll, pitch = self.get_sample(
            dataset)

        # Compute training losses.
        loss0 = self.attention_model.train(input_image, p0, p0_theta)

        loss1 = self.transport_model.train(
            input_image, p0, p1, p1_theta, z, roll, pitch)

        loss2 = self.rpz_model.train(
            input_image, p0, p1, p1_theta, z, roll, pitch)

        step = self.total_iter
        writer.add_scalars([
            ('train_loss/attention', loss0, step),
            ('train_loss/transport', loss1, step),
            ('train_loss/rpz', loss2, step),
            ('train_loss/z', self.rpz_model.z_metric.result(), step),
            ('train_loss/roll', self.rpz_model.roll_metric.result(), step),
            ('train_loss/pitch', self.rpz_model.pitch_metric.result(), step),
        ])

        print(
            f'Train Iter: {self.total_steps} \tAttention Loss: {loss0:.4f} \tTransport Loss: {loss1:.4f} \tRPZ Loss: {loss2:.4f}')

        self.total_iter += 1

    def validate(self, dataset, writer):
        """Validate on dataset.

        Args:
          dataset: a ravens_torch.Dataset.
          writer: a TensorboardX SummaryWriter.
        """

        print('Validating!')
        self.attention_model.eval_mode()()
        self.transport_model.train_mode()
        self.rpz_model.eval_mode()

        input_image, p0, p0_theta, p1, p1_theta, z, roll, pitch = self.get_data_batch(
            dataset, augment=False)

        loss0 = self.attention.test(input_image, p0, p0_theta)

        loss1 = self.transport_model.test(
            input_image, p0, p1, p1_theta, z, roll, pitch)

        loss2 = self.rpz_model.test(
            input_image, p0, p1, p1_theta, z, roll, pitch)

        # compute pixel/theta metrics
        # [metric.reset_states() for metric in self.metrics]
        # for _ in range(30):
        #     obs, act, info = validation_dataset.sample()
        #     self.act(obs, act, info, compute_error=True)

        step = self.total_steps
        writer.add_scalars([
            ('test_loss/transport', self.transport_model.metric.result(), step),
            ('test_loss/z', self.rpz_model.z_metric.result(), step),
            ('test_loss/roll', self.rpz_model.roll_metric.result(), step),
            ('test_loss/pitch', self.rpz_model.pitch_metric.result(), step),
            ('test_errors/p0_pixel_error', self.p0_pixel_error.result(), step),
            ('test_errors/p1_pixel_error', self.p1_pixel_error.result(), step),
            ('test_errors/p0_theta_error', self.p0_theta_error.result(), step),
            ('test_errors/p1_theta_error', self.p1_theta_error.result(), step),
        ])

        self.total_steps += 1
        print(
            f'Validation: \tAttention Loss: {loss0:.4f} \tTransport Loss: {loss1:.4f} \tRPZ Loss: {loss2:.4f}')

    def act(self, obs, info, compute_error=False, gt_act=None):
        """Run inference and return best action given visual observations."""

        # Get heightmap from RGB-D images.
        colormap, heightmap = self.get_heightmap(obs, self.camera_config)

        # Concatenate color with depth images.
        input_image = np.concatenate(
            (colormap, heightmap[Ellipsis, None], heightmap[Ellipsis, None], heightmap[Ellipsis,
                                                                                       None]),
            axis=2)

        # Attention model forward pass.
        attention = self.attention_model.forward(input_image)
        argmax = np.argmax(attention)
        argmax = np.unravel_index(argmax, shape=attention.shape)
        p0_pixel = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / attention.shape[2])

        # Transport model forward pass.
        transport = self.transport_model.forward(input_image, p0_pixel)
        _, z, roll, pitch = self.rpz_model.forward(input_image, p0_pixel)

        argmax = np.argmax(transport)
        argmax = np.unravel_index(argmax, shape=transport.shape)

        # Index into 3D discrete tensor, grab z, roll, pitch activations
        z_best = z[:, argmax[0], argmax[1], argmax[2]][Ellipsis, None]
        roll_best = roll[:, argmax[0], argmax[1], argmax[2]][Ellipsis, None]
        pitch_best = pitch[:, argmax[0], argmax[1], argmax[2]][Ellipsis, None]

        # Send through regressors for each of z, roll, pitch
        z_best = self.rpz_model.z_regressor(z_best)[0, 0]
        roll_best = self.rpz_model.roll_regressor(roll_best)[0, 0]
        pitch_best = self.rpz_model.pitch_regressor(pitch_best)[0, 0]

        p1_pixel = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / transport.shape[2])

        # Pixels to end effector poses.
        p0_position = utils.pix_to_xyz(p0_pixel, heightmap, self.bounds,
                                       self.pix_size)
        p1_position = utils.pix_to_xyz(p1_pixel, heightmap, self.bounds,
                                       self.pix_size)

        p1_position = (p1_position[0], p1_position[1], z_best)

        p0_rotation = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_rotation = utils.eulerXYZ_to_quatXYZW(
            (roll_best, pitch_best, -p1_theta))

        if compute_error:
            gt_p0_position, gt_p0_rotation = gt_act['params']['pose0']
            gt_p1_position, gt_p1_rotation = gt_act['params']['pose1']

            gt_p0_pixel = np.array(
                utils.xyz_to_pix(gt_p0_position, self.bounds, self.pix_size))
            gt_p1_pixel = np.array(
                utils.xyz_to_pix(gt_p1_position, self.bounds, self.pix_size))

            self.p0_pixel_error(np.linalg.norm(
                gt_p0_pixel - np.array(p0_pixel)))
            self.p1_pixel_error(np.linalg.norm(
                gt_p1_pixel - np.array(p1_pixel)))

            gt_p0_theta = -np.float32(
                utils.quatXYZW_to_eulerXYZ(gt_p0_rotation)[2])
            gt_p1_theta = -np.float32(
                utils.quatXYZW_to_eulerXYZ(gt_p1_rotation)[2])

            self.p0_theta_error(
                abs((np.rad2deg(gt_p0_theta - p0_theta) + 180) % 360 - 180))
            self.p1_theta_error(
                abs((np.rad2deg(gt_p1_theta - p1_theta) + 180) % 360 - 180))

            return None

        return {
            'pose0': (np.asarray(p0_position), np.asarray(p0_rotation)),
            'pose1': (np.asarray(p1_position), np.asarray(p1_rotation))
        }
