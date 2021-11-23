# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Resnet module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


def init_xavier_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


def Forward3LayersConvBlock(in_channels,
                            kernel_size,
                            out_channels,
                            stride=1,
                            include_batchnorm=False):

    out_channels1, out_channels2, out_channels3 = out_channels

    if include_batchnorm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels1,
                      kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels1),
            nn.ReLU(),

            nn.Conv2d(out_channels1, out_channels2,
                      kernel_size, padding=1),
            nn.BatchNorm2d(out_channels2),
            nn.ReLU(),

            nn.Conv2d(out_channels2, out_channels3, kernel_size=1),
            nn.BatchNorm2d(out_channels3),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels1,
                      kernel_size=1, stride=stride),
            nn.ReLU(),

            nn.Conv2d(out_channels1, out_channels2,
                      kernel_size, padding=1),
            nn.ReLU(),

            nn.Conv2d(out_channels2, out_channels3, kernel_size=1),
        )


class IdentityBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 out_channels,
                 activation=True,
                 include_batchnorm=False):
        """
        The identity block is the block that has no conv layer at shortcut.

        Args:
        in_channels: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        out_channels: list of integers, the filters of 3 conv layer at main path
        activation: If True, include ReLU activation on the output.
        include_batchnorm: If True, include intermediate batchnorm layers.
        """
        super().__init__()

        self.activation = activation
        self.relu = nn.ReLU()

        self.forward_block = Forward3LayersConvBlock(
            in_channels,
            kernel_size,
            out_channels,
            include_batchnorm=include_batchnorm)

    def forward(self, x):
        out = self.forward_block(x)

        out = out + x

        if self.activation:
            out = self.relu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 out_channels,
                 stride=(2, 2),
                 activation=True,
                 include_batchnorm=False):
        """A block that has a conv layer at shortcut.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well

        Args:
        in_channels: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        out_channels: list of integers, the filters of 3 conv layer at main path
        strides: Strides for the first conv layer in the block.
        activation: If True, include ReLU activation on the output.
        include_batchnorm: If True, include intermediate batchnorm layers.
        """
        super().__init__()

        self.forward_block = Forward3LayersConvBlock(
            in_channels,
            kernel_size,
            out_channels,
            stride,
            include_batchnorm)

        _, _, out_channels3 = out_channels

        self.activation = activation
        self.relu = nn.ReLU()

        self.conv_shortcut = nn.Conv2d(in_channels, out_channels3,
                                       kernel_size=1, stride=stride)

        self.include_batchnorm = include_batchnorm
        if include_batchnorm:
            self.bn_shortcut = nn.BatchNorm2d(out_channels3)

    def forward(self, x):
        out = self.forward_block(x)

        shortcut = self.conv_shortcut(x)
        if self.include_batchnorm:
            shortcut = self.bn_shortcut(shortcut)

        out = out + shortcut

        if self.activation:
            out = self.relu(out)

        return out


class ResNet43_8s(nn.Module):
    def __init__(self,
                 in_channels,
                 output_dim,
                 include_batchnorm=False,
                 cutoff_early=False):
        """Build Resnet 43 8s."""
        super().__init__()

        self.cutoff_early = cutoff_early

        if include_batchnorm:
            self.block_short = nn.Sequential(
                Rearrange('b h w c -> b c h w'),
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        else:
            self.block_short = nn.Sequential(
                Rearrange('b h w c -> b c h w'),
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        self.block_short.apply(init_xavier_weights)

        if cutoff_early:
            self.block_cutoff_early = nn.Sequential(
                ConvBlock(64, 5, [64, 64, output_dim], stride=1,
                          include_batchnorm=include_batchnorm),
                IdentityBlock(output_dim, 5, [64, 64, output_dim],
                              include_batchnorm=include_batchnorm),
                Rearrange('b c h w -> b h w c'),
            )
            self.block_cutoff_early.apply(init_xavier_weights)

        self.block_full = nn.Sequential(
            ConvBlock(64, 3, [64, 64, 64], stride=1),
            IdentityBlock(64, 3, [64, 64, 64]),

            ConvBlock(64, 3, [128, 128, 128], stride=2),
            IdentityBlock(128, 3, [128, 128, 128]),

            ConvBlock(128, 3, [256, 256, 256], stride=2),
            IdentityBlock(256, 3, [256, 256, 256]),

            ConvBlock(256, 3, [512, 512, 512], stride=2),
            IdentityBlock(512, 3, [512, 512, 512]),

            ConvBlock(512, 3, [256, 256, 256], stride=1),
            IdentityBlock(256, 3, [256, 256, 256]),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            ConvBlock(256, 3, [128, 128, 128], stride=1),
            IdentityBlock(128, 3, [128, 128, 128]),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            ConvBlock(128, 3, [64, 64, 64], stride=1),
            IdentityBlock(64, 3, [64, 64, 64]),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            ConvBlock(64, 3, [16, 16, output_dim], stride=1, activation=False),
            IdentityBlock(output_dim, 3, [
                          16, 16, output_dim], activation=False),

            Rearrange('b c h w -> b h w c'),
        )

        self.block_full.apply(init_xavier_weights)

    def forward(self, x):
        out = self.block_short(x)

        if self.cutoff_early:
            return self.block_cutoff_early(out)

        return self.block_full(out)


class ResNet36_4s(nn.Module):
    def __init__(self,
                 in_channels,
                 output_dim,
                 include_batchnorm=False,
                 cutoff_early=False):
        """Build Resnet 36 4s."""
        super().__init__()

        self.cutoff_early = cutoff_early

        if include_batchnorm:
            self.block_short = nn.Sequential(
                Rearrange('b h w c -> b c h w'),
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        else:
            self.block_short = nn.Sequential(
                Rearrange('b h w c -> b c h w'),
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        self.block_short.apply(init_xavier_weights)

        if cutoff_early:
            self.block_cutoff_early = nn.Sequential(
                ConvBlock(64, 5, [64, 64, output_dim], stride=1,
                          include_batchnorm=include_batchnorm),
                IdentityBlock(output_dim, 5, [64, 64, output_dim],
                              include_batchnorm=include_batchnorm),
                Rearrange('b c h w -> b h w c'),
            )

            self.block_cutoff_early.apply(init_xavier_weights)

        self.block_full = nn.Sequential(
            ConvBlock(64, 3, [64, 64, 64], stride=1),
            IdentityBlock(64, 3, [64, 64, 64]),

            ConvBlock(64, 3, [64, 64, 64], stride=2),
            IdentityBlock(64, 3, [64, 64, 64]),

            ConvBlock(64, 3, [64, 64, 64], stride=2),
            IdentityBlock(64, 3, [64, 64, 64]),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            ConvBlock(64, 3, [64, 64, 64], stride=1),
            IdentityBlock(64, 3, [64, 64, 64]),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            ConvBlock(64, 3, [16, 16, output_dim], stride=1, activation=False),
            IdentityBlock(output_dim, 3, [
                          16, 16, output_dim], activation=False),

            Rearrange('b c h w -> b h w c'),
        )

        self.block_full.apply(init_xavier_weights)

    def forward(self, x):
        out = self.block_short(x)

        if self.cutoff_early:
            return self.block_cutoff_early(out)

        return self.block_full(out)
