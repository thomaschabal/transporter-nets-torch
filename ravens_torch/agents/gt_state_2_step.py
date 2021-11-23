# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Ground-truth state 2-step Agent."""

import time

import numpy as np
from ravens_torch.agents.gt_state import GtState6DAgent
from ravens_torch.agents.gt_state import GtStateAgent
from ravens_torch.models import mdn_utils
from ravens_torch.models.gt_state import MlpModel
from ravens_torch.utils import utils, MeanMetrics
import torch
import torch.nn as nn
import torch.optim as optim


class GtState2StepAgent(GtStateAgent):
    """Agent which uses ground-truth state information -- useful as a baseline."""

    def __init__(self, name, task):
        super(GtState2StepAgent, self).__init__(name, task)

        # Set up model.
        self.pick_model = None
        self.place_model = None

        self.metric = MeanMetrics()
        self.val_metric = MeanMetrics()

    def init_model(self, dataset):
        """Initialize models, including normalization parameters."""
        self.set_max_obs_vector_length(dataset)

        _, _, info = dataset.random_sample()
        obs_vector = self.info_to_gt_obs(info)

        # Setup pick model
        obs_dim = obs_vector.shape[0]
        act_dim = 3
        self.pick_model = MlpModel(
            obs_dim, act_dim, 'relu', self.use_mdn, dropout=0.1)

        sampled_gt_obs = []

        num_samples = 1000
        for _ in range(num_samples):
            _, _, info = dataset.random_sample()
            t_worldaug_world, _ = self.get_augmentation_transform()
            sampled_gt_obs.append(self.info_to_gt_obs(info, t_worldaug_world))

        sampled_gt_obs = np.array(sampled_gt_obs)

        obs_train_parameters = dict()
        obs_train_parameters['mean'] = sampled_gt_obs.mean(axis=(0)).astype(
            np.float32)
        obs_train_parameters['std'] = sampled_gt_obs.std(axis=(0)).astype(
            np.float32)
        self.pick_model.set_normalization_parameters(obs_train_parameters)

        # Setup pick-conditioned place model
        obs_dim = obs_vector.shape[0] + act_dim
        act_dim = 3
        self.place_model = MlpModel(
            obs_dim, act_dim, 'relu', self.use_mdn, dropout=0.1)

        sampled_gt_obs = []

        num_samples = 1000
        for _ in range(num_samples):
            _, act, info = dataset.random_sample()
            t_worldaug_world, _ = self.get_augmentation_transform()
            obs = self.info_to_gt_obs(info, t_worldaug_world)
            obs = np.hstack(
                (obs, self.act_to_gt_act(act, t_worldaug_world)[:3]))
            sampled_gt_obs.append(obs)

        sampled_gt_obs = np.array(sampled_gt_obs)

        obs_train_parameters = dict()
        obs_train_parameters['mean'] = sampled_gt_obs.mean(axis=(0)).astype(
            np.float32)
        obs_train_parameters['std'] = sampled_gt_obs.std(axis=(0)).astype(
            np.float32)
        self.place_model.set_normalization_parameters(obs_train_parameters)

        self.pick_optim = optim.Adam(self.pick_model.parameters(), lr=2e-4)
        self.place_optim = optim.Adam(self.place_model.parameters(), lr=2e-4)

    def train(self, dataset, num_iter, writer, validation_dataset):
        """Train on dataset for a specific number of iterations."""

        if self.pick_model is None:
            self.init_model(dataset)

        if self.use_mdn:
            loss_criterion = mdn_utils.mdn_loss
        else:
            loss_criterion = nn.MSELoss()

        def train_step(pick_model, place_model, batch_obs, batch_act,
                       loss_criterion):
            self.pick_optim.zero_grad()

            prediction = pick_model(batch_obs)
            loss0 = loss_criterion(batch_act[:, 0:3], prediction)

            loss0.backward()
            self.pick_optim.step()

            self.place_optim.zero_grad()
            batch_obs = torch.cat((batch_obs, batch_act[:, 0:3]), dim=1)
            prediction = place_model(batch_obs)
            loss1 = loss_criterion(batch_act[:, 3:], prediction)

            loss1.backward()
            self.place_optim.step()

            return loss0 + loss1

        print_rate = 100
        for i in range(num_iter):
            start = time.time()

            batch_obs, batch_act, _, _, _ = self.get_data_batch(dataset)

            # Forward through model, compute training loss, update weights.
            self.metric.reset_states()
            loss = train_step(self.pick_model, self.place_model, batch_obs, batch_act,
                              loss_criterion)
            self.metric(loss)
            writer.add_scalar(
                'gt_state_loss', self.metric.result(), step=self.total_iter + i)

            if i % print_rate == 0:
                loss = np.float32(loss)
                print(f'Train Iter: {self.total_iter + i} Loss: {loss:.4f} Iter time:',
                      time.time() - start)
                # utils.meshcat_visualize(self.vis, obs, act, info)

        self.total_iter += num_iter
        self.save()

    def act(self, obs, info):
        """Run inference and return best action."""
        act = {'camera_config': self.camera_config, 'primitive': None}

        # Get observations and run pick prediction
        gt_obs = self.info_to_gt_obs(info)
        pick_prediction = self.pick_model(gt_obs[None, Ellipsis])
        if self.use_mdn:
            pi, mu, var = pick_prediction
            # prediction = mdn_utils.pick_max_mean(pi, mu, var)
            pick_prediction = mdn_utils.sample_from_pdf(pi, mu, var)
            pick_prediction = pick_prediction[:, 0, :]
        pick_prediction = pick_prediction[0]  # unbatch

        # Get observations and run place prediction
        obs_with_pick = np.hstack((gt_obs, pick_prediction))

        # since the pick at train time is always 0.0,
        # the predictions are unstable if not exactly 0
        obs_with_pick[-1] = 0.0

        place_prediction = self.place_model(obs_with_pick[None, Ellipsis])
        if self.use_mdn:
            pi, mu, var = place_prediction
            # prediction = mdn_utils.pick_max_mean(pi, mu, var)
            place_prediction = mdn_utils.sample_from_pdf(pi, mu, var)
            place_prediction = place_prediction[:, 0, :]
        place_prediction = place_prediction[0]

        prediction = np.hstack((pick_prediction, place_prediction))

        # just go exactly to objects, from observations
        # p0_position = np.hstack((gt_obs[3:5], 0.02))
        # p0_rotation = utils.eulerXYZ_to_quatXYZW(
        #     (0, 0, -gt_obs[5]*self.theta_scale))
        # p1_position = np.hstack((gt_obs[0:2], 0.02))
        # p1_rotation = utils.eulerXYZ_to_quatXYZW(
        #     (0, 0, -gt_obs[2]*self.theta_scale))

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

    def load(self, num_iter):
        """Load something."""

        # Do something here.
        # self.model.load(os.path.join(self.models_dir, model_fname))
        # Update total training iterations of agent.
        self.total_iter = num_iter

    def save(self):
        """Save models."""
        # Do something here.
        # self.model.save(os.path.join(self.models_dir, model_fname))
        pass


class GtState3Step6DAgent(GtState6DAgent):
    """Agent which uses ground-truth state information -- useful as a baseline."""

    def __init__(self, name, task):
        super().__init__(name, task)

        # Set up model.
        self.pick_model = None
        self.place_se2_model = None
        self.place_rpz_model = None

        self.metric = MeanMetrics()
        self.val_metric = MeanMetrics()

    def init_model(self, dataset):
        """Initialize models, including normalization parameters."""
        self.set_max_obs_vector_length(dataset)

        _, _, info = dataset.random_sample()
        obs_vector = self.info_to_gt_obs(info)

        # Setup pick model
        obs_dim = obs_vector.shape[0]
        act_dim = 3
        self.pick_model = MlpModel(
            obs_dim, act_dim, 'relu', self.use_mdn, dropout=0.1)

        sampled_gt_obs = []

        num_samples = 1000
        for _ in range(num_samples):
            _, _, info = dataset.random_sample()
            t_worldaug_world, _ = self.get_augmentation_transform()
            sampled_gt_obs.append(self.info_to_gt_obs(info, t_worldaug_world))

        sampled_gt_obs = np.array(sampled_gt_obs)

        obs_train_parameters = dict()
        obs_train_parameters['mean'] = sampled_gt_obs.mean(axis=(0)).astype(
            np.float32)
        obs_train_parameters['std'] = sampled_gt_obs.std(axis=(0)).astype(
            np.float32)
        self.pick_model.set_normalization_parameters(obs_train_parameters)

        # Setup pick-conditioned place se2 model
        obs_dim = obs_vector.shape[0] + act_dim
        act_dim = 3
        self.place_se2_model = MlpModel(
            obs_dim, act_dim, 'relu', self.use_mdn, dropout=0.1)

        sampled_gt_obs = []

        num_samples = 1000
        for _ in range(num_samples):
            _, act, info = dataset.random_sample()
            t_worldaug_world, _ = self.get_augmentation_transform()
            obs = self.info_to_gt_obs(info, t_worldaug_world)
            obs = np.hstack(
                (obs, self.act_to_gt_act(act, t_worldaug_world)[:3]))
            sampled_gt_obs.append(obs)

        sampled_gt_obs = np.array(sampled_gt_obs)

        obs_train_parameters = dict()
        obs_train_parameters['mean'] = sampled_gt_obs.mean(axis=(0)).astype(
            np.float32)
        obs_train_parameters['std'] = sampled_gt_obs.std(axis=(0)).astype(
            np.float32)
        self.place_se2_model.set_normalization_parameters(obs_train_parameters)

        # Setup pick-conditioned place rpz model
        obs_dim = obs_vector.shape[0] + act_dim + 3
        act_dim = 3
        self.place_rpz_model = MlpModel(
            obs_dim, act_dim, 'relu', self.use_mdn, dropout=0.1)

        sampled_gt_obs = []

        num_samples = 1000
        for _ in range(num_samples):
            _, act, info = dataset.random_sample()
            t_worldaug_world, _ = self.get_augmentation_transform()
            obs = self.info_to_gt_obs(info, t_worldaug_world)
            obs = np.hstack(
                (obs, self.act_to_gt_act(act, t_worldaug_world)[:3]))
            sampled_gt_obs.append(obs)

        sampled_gt_obs = np.array(sampled_gt_obs)

        obs_train_parameters = dict()
        obs_train_parameters['mean'] = sampled_gt_obs.mean(axis=(0)).astype(
            np.float32)
        obs_train_parameters['std'] = sampled_gt_obs.std(axis=(0)).astype(
            np.float32)
        self.place_rpz_model.set_normalization_parameters(obs_train_parameters)

        self.pick_optim = optim.Adam(self.pick_model.parameters(), lr=2e-4)
        self.place_se2_optim = optim.Adam(
            self.place_se2_model.parameters(), lr=2e-4)
        self.place_rpz_optim = optim.Adam(
            self.place_rpz_model.parameters(), lr=2e-4)

    def train(self, dataset, num_iter, writer, validation_dataset):
        """Train on dataset for a specific number of iterations."""

        if self.pick_model is None:
            self.init_model(dataset)

        if self.use_mdn:
            loss_criterion = mdn_utils.mdn_loss
        else:
            loss_criterion = nn.MSELoss()

        def train_step(pick_model, place_se2_model, place_rpz_model, batch_obs,
                       batch_act, loss_criterion):
            self.pick_optim.zero_grad()
            prediction = pick_model(batch_obs)
            loss0 = loss_criterion(batch_act[:, 0:3], prediction)
            loss0.backward()
            self.pick_optim.step()

            self.place_se2_optim.zero_grad()
            batch_obs = torch.cat((batch_obs, batch_act[:, 0:3]), dim=1)
            prediction = place_se2_model(batch_obs)
            loss1 = loss_criterion(batch_act[:, 3:6], prediction)
            loss1.backward()
            self.place_se2_optim.step()

            self.place_rpz_optim.zero_grad()
            batch_obs = torch.cat((batch_obs, batch_act[:, 3:6]), dim=1)
            prediction = place_rpz_model(batch_obs)
            loss2 = loss_criterion(batch_act[:, 6:], prediction)
            loss2.backward()
            self.place_rpz_optim.step()

            return loss0 + loss1 + loss2

        print_rate = 100
        for i in range(num_iter):
            start = time.time()

            batch_obs, batch_act, _, _, _ = self.get_data_batch(dataset)

            # Forward through model, compute training loss, update weights.
            self.metric.reset_states()
            loss = train_step(self.pick_model, self.place_se2_model,
                              self.place_rpz_model, batch_obs, batch_act,
                              loss_criterion)
            self.metric(loss)

            writer.add_scalar(
                'gt_state_loss', self.metric.result(), step=self.total_iter + i)

            if i % print_rate == 0:
                loss = np.float32(loss)
                print(f'Train Iter: {self.total_iter + i} Loss: {loss:.4f} Iter time:',
                      time.time() - start)
                # utils.meshcat_visualize(self.vis, obs, act, info)

        self.total_iter += num_iter
        self.save()

    def act(self, obs, info):
        """Run inference and return best action."""
        act = {'camera_config': self.camera_config, 'primitive': None}

        # Get observations and run pick prediction
        gt_obs = self.info_to_gt_obs(info)
        pick_prediction = self.pick_model(gt_obs[None, Ellipsis])
        if self.use_mdn:
            pi, mu, var = pick_prediction
            # prediction = mdn_utils.pick_max_mean(pi, mu, var)
            pick_prediction = mdn_utils.sample_from_pdf(pi, mu, var)
            pick_prediction = pick_prediction[:, 0, :]
        pick_prediction = pick_prediction[0]  # unbatch

        # Get observations and run place prediction
        obs_with_pick = np.hstack((gt_obs, pick_prediction)).astype(np.float32)

        # since the pick at train time is always 0.0,
        # the predictions are unstable if not exactly 0
        obs_with_pick[-1] = 0.0

        place_se2_prediction = self.place_se2_model(
            obs_with_pick[None, Ellipsis])
        if self.use_mdn:
            pi, mu, var = place_se2_prediction
            # prediction = mdn_utils.pick_max_mean(pi, mu, var)
            place_se2_prediction = mdn_utils.sample_from_pdf(pi, mu, var)
            place_se2_prediction = place_se2_prediction[:, 0, :]
        place_se2_prediction = place_se2_prediction[0]

        # Get observations and run rpz prediction
        obs_with_pick_place_se2 = np.hstack(
            (obs_with_pick, place_se2_prediction)).astype(np.float32)

        place_rpz_prediction = self.place_rpz_model(obs_with_pick_place_se2[None,
                                                                            Ellipsis])
        if self.use_mdn:
            pi, mu, var = place_rpz_prediction
            # prediction = mdn_utils.pick_max_mean(pi, mu, var)
            place_rpz_prediction = mdn_utils.sample_from_pdf(pi, mu, var)
            place_rpz_prediction = place_rpz_prediction[:, 0, :]
        place_rpz_prediction = place_rpz_prediction[0]

        p0_position = np.hstack((pick_prediction[0:2], 0.02))
        p0_rotation = utils.eulerXYZ_to_quatXYZW((0, 0, 0))

        p1_position = np.hstack(
            (place_se2_prediction[0:2], place_rpz_prediction[2]))
        p1_rotation = utils.eulerXYZ_to_quatXYZW(
            (place_rpz_prediction[0] * self.theta_scale,
             place_rpz_prediction[1] * self.theta_scale,
             -place_se2_prediction[2] * self.theta_scale))

        # Select task-specific motion primitive.
        act['primitive'] = 'pick_place_6dof'

        params = {
            'pose0': (p0_position, p0_rotation),
            'pose1': (p1_position, p1_rotation)
        }
        act['params'] = params
        return act
