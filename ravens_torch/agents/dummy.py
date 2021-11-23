# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Dummy Agent."""

import os

import numpy as np

from ravens_torch.tasks import cameras
from ravens_torch.utils import utils


class DummyAgent:
    """Dummy Agent."""

    def __init__(self, name, task, root_dir, verbose=False):
        self.name = name
        self.task = task
        self.total_steps = 0

        # Share same camera configuration as Transporter.
        self.camera_config = cameras.RealSenseD415.CONFIG

        # [Optional] Heightmap parameters.
        self.pixel_size = 0.003125
        self.bounds = np.float32([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        # A place to save pre-trained models.
        self.models_dir = os.path.join(root_dir, 'checkpoints', self.name)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def train(self, dataset, writer):
        """Train on dataset for a specific number of iterations."""
        del writer

        (obs, act, _, _), _ = dataset.sample()

        # Do something here.

        # Compute training loss here.
        loss = 0.
        print(f'Train Iter: {self.total_steps} Loss: {loss:.4f}')

        self.total_steps += 1

    def validate(self, dataset, writer):
        """Train on dataset for a specific number of iterations."""
        del writer

        (obs, act, _, _), _ = dataset.sample()

        # Do something here.

        # Compute training loss here.
        loss = 0.
        print(f'Validation Iter: {self.total_steps} Loss: {loss:.4f}')

        self.total_steps += 1

    def act(self, obs, info, goal):
        """Run inference and return best action given visual observations."""
        del info

        act = {'camera_config': self.camera_config, 'primitive': None}
        if not obs:
            return act

        # [Optional] Get heightmap from RGB-D images.
        colormap, heightmap = self.get_heightmap(obs, self.camera_config)

        # Do something here.

        # Dummy behavior: move to the middle of the workspace.
        p0_position = (self.bounds[:, 1] - self.bounds[:, 0]) / 2
        p0_position += self.bounds[:, 0]
        p1_position = p0_position
        rotation = utils.eulerXYZ_to_quatXYZW((0, 0, 0))

        # Select task-specific motion primitive.
        act['primitive'] = 'pick_place'
        if self.task == 'sweeping':
            act['primitive'] = 'sweep'
        elif self.task == 'pushing':
            act['primitive'] = 'push'

        params = {
            'pose0': (np.asarray(p0_position), np.asarray(rotation)),
            'pose1': (np.asarray(p1_position), np.asarray(rotation))
        }
        act['params'] = params
        return params
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

    def preprocess(self, image):
        """Pre-process images (subtract mean, divide by std)."""
        color_mean = 0.18877631
        depth_mean = 0.00509261
        color_std = 0.07276466
        depth_std = 0.00903967
        image[:, :, :3] = (image[:, :, :3] / 255 - color_mean) / color_std
        image[:, :, 3:] = (image[:, :, 3:] - depth_mean) / depth_std
        return image

    def get_heightmap(self, obs, configs):
        """Reconstruct orthographic heightmaps with segmentation masks."""
        heightmaps, colormaps = utils.reconstruct_heightmaps(
            obs['color'], obs['depth'], configs, self.bounds, self.pixel_size)
        colormaps = np.float32(colormaps)
        heightmaps = np.float32(heightmaps)

        # Fuse maps from different views.
        valid = np.sum(colormaps, axis=3) > 0
        repeat = np.sum(valid, axis=0)
        repeat[repeat == 0] = 1
        colormap = np.sum(colormaps, axis=0) / repeat[Ellipsis, None]
        colormap = np.uint8(np.round(colormap))
        heightmap = np.sum(heightmaps, axis=0) / repeat

        return colormap, heightmap
