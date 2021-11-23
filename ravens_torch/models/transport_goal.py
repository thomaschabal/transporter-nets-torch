# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Goal-conditioned transport Module."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ravens_torch.models.resnet import ResNet43_8s
from ravens_torch.utils import utils, MeanMetrics, to_device
from ravens_torch.utils.utils import apply_rotations_to_tensor


class TransportGoal:
    """Goal-conditioned transport Module."""

    def __init__(self, in_channels, num_rotations, crop_size, preprocess, verbose=False):  # pylint: disable=g-doc-args
        """Inits transport module with separate goal FCN.

        Assumes the presence of a goal image, that cropping is done after the
        query, that per-pixel loss is not used, and SE(2) grasping.
        """
        self.num_rotations = num_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess
        self.lr = 1e-5

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        # Output dimension (i.e., number of channels) of 3.
        self.odim = output_dim = 3

        # 3 fully convolutional ResNets. Third one is for the goal.
        self.model_logits = ResNet43_8s(in_channels, output_dim)
        self.model_kernel = ResNet43_8s(in_channels, output_dim)
        self.model_goal = ResNet43_8s(in_channels, output_dim)

        self.device = to_device(
            [self.model_logits, self.model_kernel, self.model_goal],
            "Transport Goal",
            verbose=verbose)

        self.optimizer_logits = optim.Adam(
            self.model_logits.parameters(), lr=self.lr)
        self.optimizer_kernel = optim.Adam(
            self.model_kernel.parameters(), lr=self.lr)
        self.optimizer_goal = optim.Adam(
            self.model_goal.parameters(), lr=self.lr)

        self.metric = MeanMetrics()
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_img, goal_img, p, apply_softmax=True):  # pylint: disable=g-doc-args
        """Forward pass of goal-conditioned Transporters.

        Runs input through all three networks, to get output of the same
        shape, except the last channel is 3 (output_dim). Then, the output
        for one stream has the convolutional kernels for another. Call
        torch.nn.functional.conv2d, and the operation is be differentiable, so that
        gradients apply to all the FCNs.

        Cropping after passing the input image to the query network is
        easier, because otherwise we need to do a forward pass, then call
        torch.multiply, then do a second forward pass after that.

        Returns:
          ouput tensor
        """
        assert in_img.shape == goal_img.shape, f'{in_img.shape}, {goal_img.shape}'

        # input image --> Torch tensor, shape (384,224,6) --> (1,384,224,6)
        input_unproc = np.pad(in_img, self.padding, mode='constant')
        input_data = self.preprocess(input_unproc.copy())
        input_data = Rearrange('h w c -> 1 h w c')(input_data)
        in_tensor = torch.tensor(
            input_data, dtype=torch.float32
        ).to(self.device)

        # goal image --> Torch tensor, shape (384,224,6) --> (1,384,224,6)
        goal_unproc = np.pad(goal_img, self.padding, mode='constant')
        goal_data = self.preprocess(goal_unproc.copy())
        goal_data = Rearrange('h w c -> 1 h w c')(goal_data)
        goal_tensor = torch.tensor(
            goal_data, dtype=torch.float32
        ).to(self.device)

        # Get SE2 rotation vectors for cropping.
        pivot = np.array([p[1], p[0]]) + self.pad_size

        # Forward pass through three separate FCNs. All logits: (1,384,224,3).
        in_logits = self.model_logits(in_tensor)
        kernel_nocrop_logits = self.model_kernel(in_tensor)
        goal_logits = self.model_goal(goal_tensor)

        # Use features from goal logits and combine with input and kernel.
        # CHECK CODE FOR MULTIPLIES
        goal_x_in_logits = torch.multiply(goal_logits, in_logits)
        goal_x_kernel_logits = torch.multiply(
            goal_logits, kernel_nocrop_logits)

        # Crop the kernel_logits about the picking point and get rotations.
        crop = apply_rotations_to_tensor(
            goal_x_kernel_logits, self.num_rotations, center=pivot)
        kernel = crop[:, p[0]:(p[0] + self.crop_size),
                      p[1]:(p[1] + self.crop_size), :]
        assert kernel.shape == (self.num_rotations, self.crop_size, self.crop_size,
                                self.odim)

        # Cross-convolve `in_x_goal_logits`. Padding kernel: (24,64,64,3) -->
        # (24,3,65,65).
        kernel_paddings = nn.ConstantPad2d((0, 0, 0, 1, 0, 1, 0, 0), 0)
        kernel = kernel_paddings(kernel)

        goal_x_in_logits = Rearrange('b h w c -> b c h w')(goal_x_in_logits)
        kernel = Rearrange('b h w c -> b c h w')(kernel)

        output = F.conv2d(goal_x_in_logits, kernel)
        output = output / (self.crop_size ** 2)

        if apply_softmax:
            output_shape = output.shape
            output = Rearrange('b c h w -> b (c h w)')(output)
            output = self.softmax(output)
            output = Rearrange(
                'b (c h w) -> b c h w',
                c=output_shape[1],
                h=output_shape[2],
                w=output_shape[3])(output)
            output = output.detach().cpu().numpy()

        # Daniel: visualize crops and kernels, for Transporter-Goal figure.
        # self.visualize_images(p, in_img, input_data, crop)
        # self.visualize_transport(p, in_img, input_data, crop, kernel)
        # self.visualize_logits(in_logits,            name='input')
        # self.visualize_logits(goal_logits,          name='goal')
        # self.visualize_logits(kernel_nocrop_logits, name='kernel')
        # self.visualize_logits(goal_x_in_logits,     name='goal_x_in')
        # self.visualize_logits(goal_x_kernel_logits, name='goal_x_kernel')

        return output

    def train(self, in_img, goal_img, p, q, theta):
        """Transport Goal training.

        Both `in_img` and `goal_img` have the color and depth. Much is
        similar to the attention model: (a) forward pass, (b) get angle
        discretizations, (c) make the label consider rotations in the last
        axis, but only provide the label to one single (pixel,rotation).

        Args:
          in_img:
          goal_img:
          p:
          q:
          theta:

        Returns:
          Transport loss as a numpy float32.
        """
        self.metric.reset()
        self.optimizer_logits.zero_grad()
        self.optimizer_kernel.zero_grad()
        self.optimizer_goal.zero_grad()

        output = self.forward(in_img, goal_img, p, apply_softmax=False)

        # Compute label
        itheta = theta / (2 * np.pi / self.num_rotations)
        itheta = np.int32(np.round(itheta)) % self.num_rotations
        label_size = in_img.shape[:2] + (self.num_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1

        label = Rearrange('h w c -> 1 (h w c)')(label)
        label = torch.tensor(label, dtype=torch.float32).to(self.device)
        label = torch.argmax(label, dim=1)

        # Compute loss after re-shaping the output.
        output = Rearrange('b theta h w -> b (h w theta)')(output)
        loss = self.loss(output, label)

        loss.backward()
        self.optimizer_logits.step()
        self.optimizer_kernel.step()
        self.optimizer_goal.step()
        self.metric(loss)

        return np.float32(loss.detach().cpu().numpy())

    def format_fname(self, fname, is_logits=False, is_kernel=False, is_goal=False):
        if is_logits:
            suffix = 'logits'
        elif is_kernel:
            suffix = 'kernel'
        elif is_goal:
            suffix = 'goal'
        else:
            raise Exception("Model to load not specified.")

        return fname.split('.pth')[0] + f'_{suffix}.pth'

    def save(self, fname):
        logits_name = self.format_fname(fname, is_logits=True)
        kernel_name = self.format_fname(fname, is_kernel=True)
        goal_name = self.format_fname(fname, is_goal=True)

        torch.save(self.model_logits.state_dict(), logits_name)
        torch.save(self.model_kernel.state_dict(), kernel_name)
        torch.save(self.model_goal.state_dict(), goal_name)

    def load(self, fname):
        logits_name = self.format_fname(fname, is_logits=True)
        kernel_name = self.format_fname(fname, is_kernel=True)
        goal_name = self.format_fname(fname, is_goal=True)

        self.model_logits.load_state_dict(torch.load(logits_name))
        self.model_kernel.load_state_dict(torch.load(kernel_name))
        self.model_goal.load_state_dict(torch.load(goal_name))

    # -------------------------------------------------------------------------
    # Visualization.
    # -------------------------------------------------------------------------

    def visualize_images(self, p, in_img, input_data, crop):
        """Visualize images."""

        def get_itheta(theta):
            itheta = theta / (2 * np.pi / self.num_rotations)
            return np.int32(np.round(itheta)) % self.num_rotations

        plt.subplot(1, 3, 1)
        plt.title('Perturbed', fontsize=15)
        plt.imshow(np.array(in_img[:, :, :3]).astype(np.uint8))
        plt.subplot(1, 3, 2)
        plt.title('Process/Pad', fontsize=15)
        plt.imshow(input_data[0, :, :, :3])
        plt.subplot(1, 3, 3)
        # Let's stack two crops together.
        theta1 = 0.0
        theta2 = 90.0
        itheta1 = get_itheta(theta1)
        itheta2 = get_itheta(theta2)
        crop1 = crop[itheta1, :, :, :3]
        crop2 = crop[itheta2, :, :, :3]
        barrier = np.ones_like(crop1)
        barrier = barrier[:4, :, :]  # white barrier of 4 pixels
        stacked = np.concatenate((crop1, barrier, crop2), axis=0)
        plt.imshow(stacked)
        plt.title(f'{theta1}, {theta2}', fontsize=15)
        plt.suptitle(f'pick: {p}', fontsize=15)
        plt.tight_layout()
        plt.show()
        # plt.savefig('viz.png')

    def visualize_transport(self, p, in_img, input_data, crop, kernel):  # pylint: disable=g-doc-args
        """Like the attention map visualize the transport data from a trained model.

        https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
        In my normal usage, the attention is already softmax-ed but just be
        aware in case it's not. Also be aware of RGB vs BGR mode. We should
        ensure we're in BGR mode before saving. Also with RAINBOW mode,
        red=hottest (highest attention values), green=medium, blue=lowest.

        See also:
        https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.pyplot.subplot.html

        crop.shape: (24,64,64,6)
        kernel.shape = (65,65,3,24)
        """
        del p
        del in_img
        del input_data

        def colorize(img):
            # I don't think we have to convert to BGR here...
            img = img - np.min(img)
            img = 255 * img / np.max(img)
            img = cv2.applyColorMap(np.uint8(img), cv2.COLORMAP_RAINBOW)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        kernel = Rearrange('b h w theta -> b theta h w')(kernel)

        # Top two rows: crops from processed RGBD. Bottom two: output from FCN.
        nrows = 4
        ncols = 12
        assert self.num_rotations == nrows * (ncols / 2)
        idx = 0
        _, _ = plt.subplots(nrows, ncols, figsize=(12, 6))
        for _ in range(nrows):
            for _ in range(ncols):
                plt.subplot(nrows, ncols, idx + 1)
                plt.axis('off')  # Ah, you need to put this here ...
                if idx < self.num_rotations:
                    plt.imshow(crop[idx, :, :, :3])
                else:
                    # Offset because idx goes from 0 to (rotations * 2) - 1.
                    idx_ = idx - self.num_rotations
                    processed = colorize(img=kernel[:, idx_, :, :])
                    plt.imshow(processed)
                idx += 1
        plt.tight_layout()
        plt.show()

    def visualize_logits(self, logits, name):  # pylint: disable=g-doc-args
        """Given logits (BEFORE tf.nn.convolution), get a heatmap.

        Here we apply a softmax to make it more human-readable. However, the
        tf.nn.convolution with the learned kernels happens without a softmax
        on the logits. [Update: wait, then why should we have a softmax,
        then? I forgot why we did this ...]
        """
        original_shape = logits.shape
        logits = Rearrange('b c h w -> b (h w c)')(logits)
        # logits = self.softmax(logits)  # Is this necessary?
        vis_transport = Rearrange('b (h w c) -> b c h w',
                                  c=original_shape[1],
                                  h=original_shape[2],
                                  w=original_shape[3])(logits).detach().cpu().numpy()
        vis_transport = vis_transport[0]
        vis_transport = vis_transport - np.min(vis_transport)
        vis_transport = 255 * vis_transport / np.max(vis_transport)
        vis_transport = cv2.applyColorMap(
            np.uint8(vis_transport), cv2.COLORMAP_RAINBOW)

        # Only if we're saving with cv2.imwrite()
        vis_transport = cv2.cvtColor(vis_transport, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'tmp/logits_{name}.png', vis_transport)

        plt.subplot(1, 1, 1)
        plt.title(f'Logits: {name}', fontsize=15)
        plt.imshow(vis_transport)
        plt.tight_layout()
        plt.show()
