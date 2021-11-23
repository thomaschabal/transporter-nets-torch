# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Regression module."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ravens_torch.models import mdn_utils
from ravens_torch.models.conv_mlp import ConvMLP, DeepConvMLP
from ravens_torch.utils import utils, MeanMetrics, to_device


def Regression(in_channels, verbose=False):
    """Regression module."""
    model = nn.Sequential(
        nn.Linear(in_channels, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    _ = to_device([model], "Regression", verbose=verbose)

    return model


# class Regression:
#     """Regression module."""

#     def __init__(self, in_channels, preprocess, use_mdn, verbose=False):
#         self.preprocess = preprocess

#         resnet = False

#         if resnet:
#             self.model = DeepConvMLP(in_channels, d_action=6, use_mdn=use_mdn)
#         else:
#             self.model = ConvMLP(d_action=6, use_mdn=use_mdn)
#         self.device = to_device([self.model], "Regression", verbose=verbose)

#         self.optim = optim.Adam(self.model.parameters(), lr=2e-4)
#         self.metric = MeanMetrics()
#         self.val_metric = MeanMetrics()

#         self.loss_criterion = nn.MSELoss() if not use_mdn else mdn_utils.mdn_loss

#     def set_batch_size(self, batch_size):
#         self.model.set_batch_size(batch_size)

#     def forward(self, in_img):
#         """Forward pass.

#         Args:
#           in_img: [B, C, H, W]

#         Returns:
#           output tensor.
#         """
#         input_data = self.preprocess(in_img)
#         in_tensor = torch.tensor(
#             input_data, dtype=torch.float32).to(self.device)
#         output = self.model(in_tensor)
#         return output

#     def train_pick(self, batch_obs, batch_act, train_step, validate=False):
#         """Train pick."""
#         self.metric.reset()
#         self.val_metric.reset()

#         input_data = self.preprocess(batch_obs)
#         in_tensor = torch.tensor(
#             input_data, dtype=torch.float32).to(self.device)

#         loss = train_step(self.model, self.optim, in_tensor, batch_act,
#                           self.loss_criterion)

#         if not validate:
#             self.metric(loss)
#         else:
#             self.val_metric(loss)
#         return np.float32(loss)

#     def save(self, fname):
#         torch.save(self.model.state_dict(), fname)

#     def load(self, fname):
#         self.model.load_state_dict(torch.load(fname))
