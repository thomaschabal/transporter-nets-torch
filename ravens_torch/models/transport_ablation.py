# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Transport ablations."""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from ravens_torch.models.transport import Transport
from ravens_torch.utils import utils


class TransportPerPixelLoss(Transport):
    """Transport + per-pixel loss ablation."""

    def __init__(self, in_channels, n_rotations, crop_size, preprocess, verbose=False):
        self.output_dim = 6
        super().__init__(in_channels, n_rotations, crop_size,
                         preprocess, verbose, name="Transport Goal")

    def correlate(self, in0, in1, softmax):
        in0 = Rearrange('b h w c -> b c h w')(in0)
        in1 = Rearrange('b h w c -> b c h w')(in1)

        output0 = F.conv2d(in0[:, :3, ...], in1)
        output1 = F.conv2d(in0[:, 3:, ...], in1)
        output = torch.cat((output0, output1), dim=0)  # (2,c,h,w)

        if softmax:
            output_shape = output.shape
            output = Rearrange('b c h w -> b (c h w)')(output)
            output = self.softmax(output)
            output = Rearrange(
                'b (c h w) -> b h w c',
                c=output_shape[1],
                h=output_shape[2],
                w=output_shape[3])(output)  # (2,h,w,c)
            output = output[1, ...].detach().cpu().numpy()  # (h,w,c)
        return output

    def train_block(self, in_img, p, q, theta):
        output = self.forward(in_img, p, softmax=False)

        itheta = theta / (2 * np.pi / self.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.n_rotations

        # Get one-hot pixel label map.
        label_size = in_img.shape[:2] + (self.n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1

        # Get per-pixel sampling loss.
        sampling = True  # Sampling negatives seems to converge faster.
        if sampling:
            num_samples = 100
            inegative = utils.sample_distribution(1 - label, num_samples)
            inegative = [np.ravel_multi_index(
                i, label.shape) for i in inegative]
            ipositive = np.ravel_multi_index([q[0], q[1], itheta], label.shape)

            # BEWARE OF POSSIBLE TROUBLES DUE TO THE ORDER OF INDICES
            output = Rearrange('b c h w -> b (h w c)')(output)
            output_samples = ()
            for i in inegative:
                out_sample = output[:, i][np.newaxis, ...]  # (1,2)
                output_samples += (out_sample,)

            output_samples += (output[:, ipositive][np.newaxis, ...],)  # (1,2)
            output = torch.cat(output_samples, dim=0)  # (num_samples+1, 2)

            label = np.int32([0] * num_samples + [1])[Ellipsis, None]
            label = np.hstack((1 - label, label))
            weights = np.ones(label.shape[0])
            weights[:num_samples] = 1. / num_samples
            weights = weights / np.sum(weights)

        else:
            ipositive = np.ravel_multi_index([q[0], q[1], itheta], label.shape)
            output = Rearrange('b c h w -> b (h w c)')(output)
            label = np.int32(np.reshape(label, (int(np.prod(label.shape)), 1)))
            label = np.hstack((1 - label, label))
            weights = np.ones(label.shape[0]) * 0.0025  # Magic constant.
            weights[ipositive] = 1

        # Get loss.
        label = torch.tensor(label, dtype=torch.int32).to(self.device)
        label = torch.argmax(label, dim=1)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
        loss = self.loss(output, label)
        loss = torch.mean(torch.multiply(loss, weights))

        return loss

    def train(self, in_img, p, q, theta, backprop=True):
        self.metric.reset()
        self.model_query.train()
        self.model_key.train()
        self.optimizer_query.zero_grad()
        self.optimizer_key.zero_grad()

        loss = self.train_block(in_img, p, q, theta)

        if backprop:
            loss.backward()
            self.optimizer_query.step()
            self.optimizer_key.step()
        self.metric(loss)

        self.iters += 1
        return np.float32(loss.detach().cpu().numpy())

    def test(self, in_img, p, q, theta):
        self.model_query.eval()
        self.model_key.eval()

        with torch.no_grad():
            loss = self.train_block(in_img, p, q, theta)

        self.metric(loss)

        self.iters += 1
        return np.float32(loss.detach().cpu().numpy())
