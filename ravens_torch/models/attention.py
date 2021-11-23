# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Attention module."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ravens_torch.utils import utils, MeanMetrics, to_device
from ravens_torch.utils.text import bold
from ravens_torch.utils.utils import apply_rotations_to_tensor
from ravens_torch.models.resnet import ResNet43_8s, ResNet36_4s

from einops.layers.torch import Rearrange

# REMOVE BELOW
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


class Attention:
    """Attention module."""

    def __init__(self, in_shape, n_rotations, preprocess, lite=False, verbose=False):
        self.n_rotations = n_rotations
        self.preprocess = preprocess

        max_dim = np.max(in_shape[:2])

        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        # Initialize fully convolutional Residual Network with 43 layers and
        # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
        model_type = ResNet36_4s if lite else ResNet43_8s
        self.model = model_type(in_shape[2], 1)

        self.device = to_device([self.model], "Attention", verbose=verbose)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss = nn.CrossEntropyLoss(reduction="mean")

        self.metric = MeanMetrics()

    def forward(self, in_img, softmax=True):
        """Forward pass."""
        in_data = np.pad(in_img, self.padding, mode='constant')
        in_data = self.preprocess(in_data)
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.tensor(in_data, dtype=torch.float32).to(self.device)

        # Rotate input.
        in_tens = apply_rotations_to_tensor(in_tens, self.n_rotations)

        # Forward pass.
        in_tens = torch.split(in_tens, 1, dim=0)  # (self.num_rotations)
        logits = ()
        for x in in_tens:
            logits += (self.model(x),)
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = apply_rotations_to_tensor(
            logits, self.n_rotations, reverse=True)

        c0 = self.padding[:2, 0]
        c1 = c0 + in_img.shape[:2]
        logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]

        output = Rearrange('b h w c -> b (h w c)')(logits)

        if softmax:
            output = nn.Softmax(dim=1)(output)
            output = output.detach().cpu().numpy()
            output = np.float32(output).reshape(logits.shape[1:])
        return output

    def train_block(self, in_img, p, theta):
        output = self.forward(in_img, softmax=False)

        # Get label.
        theta_i = theta / (2 * np.pi / self.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.n_rotations
        label_size = in_img.shape[:2] + (self.n_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        label = torch.tensor(label, dtype=torch.float32).to(self.device)

        # Get loss.
        label = Rearrange('h w c -> 1 (h w c)')(label)
        label = torch.argmax(label, dim=1)

        loss = self.loss(output, label)

        return loss

    def train(self, in_img, p, theta):
        """Train."""
        self.metric.reset()
        self.train_mode()
        self.optimizer.zero_grad()

        loss = self.train_block(in_img, p, theta)
        loss.backward()
        self.optimizer.step()
        self.metric(loss)

        return np.float32(loss.detach().cpu().numpy())

    def test(self, in_img, p, theta):
        """Test."""
        self.eval_mode()

        with torch.no_grad():
            loss = self.train_block(in_img, p, theta)

        return np.float32(loss.detach().cpu().numpy())

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def load(self, path, verbose=False):
        if verbose:
            device = "GPU" if self.device.type == "cuda" else "CPU"
            print(
                f"Loading {bold('attention')} model on {bold(device)} from {bold(path)}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, filename, verbose=False):
        if verbose:
            print(f"Saving attention model to {bold(filename)}")
        torch.save(self.model.state_dict(), filename)
