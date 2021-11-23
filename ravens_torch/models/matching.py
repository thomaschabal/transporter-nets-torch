# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Matching module."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from ravens_torch.models.resnet import ResNet43_8s, ResNet36_4s
from ravens_torch.utils import utils, MeanMetrics, ContrastiveLoss, to_device
from ravens_torch.utils.utils import apply_rotations_to_tensor


class Matching:
    """Matching module."""

    def __init__(self,
                 input_shape,
                 descriptor_dim,
                 num_rotations,
                 preprocess,
                 lite=False,
                 verbose=False):
        self.preprocess = preprocess
        self.num_rotations = num_rotations
        self.descriptor_dim = descriptor_dim

        max_dim = np.max(input_shape[:2])

        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(input_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        # Initialize fully convolutional Residual Network with 43 layers and
        # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
        model_type = ResNet36_4s if lite else ResNet43_8s
        self.model = model_type(input_shape[2], self.descriptor_dim)

        self.device = to_device([self.model], "Matching", verbose=verbose)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

        self.metric = MeanMetrics()
        self.loss = ContrastiveLoss()

    def forward(self, input_image):
        """Forward pass."""
        input_data = np.pad(input_image, self.padding, mode='constant')
        input_data = self.preprocess(input_data)
        input_shape = (1,) + input_data.shape
        input_data = input_data.reshape(input_shape)
        input_tensor = torch.tensor(
            input_data, dtype=torch.float32).to(self.device)

        # Rotate input.
        input_tensor = apply_rotations_to_tensor(
            input_tensor, self.num_rotations)

        # Forward pass.
        input_tensor = torch.split(input_tensor, self.num_rotations, dim=0)
        logits = ()
        for x in input_tensor:
            logits += (self.model(x),)
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = apply_rotations_to_tensor(
            logits, self.num_rotations, reverse=True)

        c0 = self.padding[:2, 0]
        c1 = c0 + input_image.shape[:2]
        output = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]

        return output

    def compute_loss(self, input_image, output, p, q, theta):
        p_descriptor = output[:, p[0], p[1], 0]
        itheta = theta / (2 * np.pi / self.num_rotations)
        itheta = np.int32(np.round(itheta)) % self.num_rotations
        q_descriptor = output[:, q[0], q[1], itheta]

        # Positives.
        positive_distances = torch.linalg.norm(p_descriptor - q_descriptor)
        positive_distances = Rearrange('b -> (b)')(positive_distances)
        positive_labels = torch.tensor([1], dtype=torch.int32)
        positive_loss = self.loss(positive_distances, positive_labels)

        # Negatives.
        num_samples = 100
        sample_map = np.zeros(input_image.shape[:2] + (self.num_rotations,))
        sample_map[p[0], p[1], 0] = 1
        sample_map[q[0], q[1], itheta] = 1
        inegative = utils.sample_distribution(1 - sample_map, num_samples)
        negative_distances = ()
        negative_labels = ()
        for i in range(num_samples):
            descriptor = output[
                inegative[i, 2],
                inegative[i, 0],
                inegative[i, 1],
                :
            ]
            distance = torch.linalg.norm(p_descriptor - descriptor)
            distance = Rearrange('b -> (b)')(distance)
            negative_distances += (distance,)
            negative_labels += (torch.tensor([0], dtype=torch.int32),)
        negative_distances = torch.cat(negative_distances, dim=0)
        negative_labels = torch.cat(negative_labels, dim=0)
        negative_loss = self.loss(negative_distances, negative_labels)

        loss = positive_loss + negative_loss

        # loss = tf.reduce_mean(positive_loss) + \
        #     tf.reduce_mean(negative_loss)

        return loss

    def train(self, input_image, p, q, theta):
        """Train function."""
        self.metric.reset()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.forward(input_image)

        loss = self.compute_loss(input_image, output, p, q, theta)

        # Backpropagate.
        loss.backward()
        self.optimizer.step()

        self.metric(loss)
        return np.float32(loss)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def get_se2(self, num_rotations, reverse=False):
        """Get SE2 rotations discretized into num_rotations angles counter-clockwise."""
        thetas = []
        for i in range(num_rotations):
            theta = i * 360 / num_rotations
            theta = -theta if reverse else theta
            thetas.append(theta)
        return np.array(thetas)
