# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Transporter Agent."""

import os

import numpy as np
from ravens_torch.models.attention import Attention
from ravens_torch.models.transport import Transport
from ravens_torch.models.transport_ablation import TransportPerPixelLoss
from ravens_torch.models.transport_goal import TransportGoal
from ravens_torch.tasks import cameras
from ravens_torch.utils import utils


class TransporterAgent:
    """Agent that uses Transporter Networks."""

    def __init__(self, name, task, root_dir, n_rotations=36):
        self.name = name
        self.task = task
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = n_rotations
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.models_dir = os.path.join(root_dir, 'checkpoints', self.name)
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

    def get_image(self, obs):
        """Stack color and height images image."""

        # if self.use_goal_image:
        #   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
        #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
        #   input_image = np.concatenate((input_image, goal_image), axis=2)
        #   assert input_image.shape[2] == 12, input_image.shape

        # Get color and height maps from RGB-D images.
        cmap, hmap = utils.get_fused_heightmap(
            obs, self.cam_config, self.bounds, self.pix_size)
        img = np.concatenate((cmap,
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None]), axis=2)
        assert img.shape == self.in_shape, img.shape
        return img

    def get_sample(self, dataset, augment=True):
        """Get a dataset sample.

        Args:
          dataset: a ravens_torch.Dataset (train or validation)
          augment: if True, perform data augmentation.

        Returns:
          tuple of data for training:
            (input_image, p0, p0_theta, p1, p1_theta)
          tuple additionally includes (z, roll, pitch) if self.six_dof
          if self.use_goal_image, then the goal image is stacked with the
          current image in `input_image`. If splitting up current and goal
          images is desired, it should be done outside this method.
        """

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

        # Data augmentation.
        if augment:
            img, _, (p0, p1), _ = utils.perturb(img, [p0, p1])

        return img, p0, p0_theta, p1, p1_theta

    def train(self, dataset, writer=None):
        """Train on a dataset sample for 1 iteration.

        Args:
          dataset: a ravens_torch.Dataset.
          writer: a TensorboardX SummaryWriter.
        """
        self.attention.train_mode()
        self.transport.train_mode()

        img, p0, p0_theta, p1, p1_theta = self.get_sample(dataset)

        # Get training losses.
        step = self.total_steps + 1
        loss0 = self.attention.train(img, p0, p0_theta)
        if isinstance(self.transport, Attention):
            loss1 = self.transport.train(img, p1, p1_theta)
        else:
            loss1 = self.transport.train(img, p0, p1, p1_theta)

        writer.add_scalars([
            ('train_loss/attention', loss0, step),
            ('train_loss/transport', loss1, step),
        ])

        print(
            f'Train Iter: {step} \t Attention Loss: {loss0:.4f} \t Transport Loss: {loss1:.4f}')
        self.total_steps = step

        # TODO(andyzeng) cleanup goal-conditioned model.

        # if self.use_goal_image:
        #   half = int(input_image.shape[2] / 2)
        #   img_curr = input_image[:, :, :half]  # ignore goal portion
        #   loss0 = self.attention.train(img_curr, p0, p0_theta)
        # else:
        #   loss0 = self.attention.train(input_image, p0, p0_theta)

        # if isinstance(self.transport, Attention):
        #   loss1 = self.transport.train(input_image, p1, p1_theta)
        # elif isinstance(self.transport, TransportGoal):
        #   half = int(input_image.shape[2] / 2)
        #   img_curr = input_image[:, :, :half]
        #   img_goal = input_image[:, :, half:]
        #   loss1 = self.transport.train(img_curr, img_goal, p0, p1, p1_theta)
        # else:
        #   loss1 = self.transport.train(input_image, p0, p1, p1_theta)

    def validate(self, dataset, writer=None):  # pylint: disable=unused-argument
        """Test on a validation dataset for 10 iterations."""

        n_iter = 10
        loss0, loss1 = 0, 0
        for _ in range(n_iter):
            img, p0, p0_theta, p1, p1_theta = self.get_sample(dataset, False)

            # Get validation losses. Do not backpropagate.
            loss0 += self.attention.test(img, p0, p0_theta)
            if isinstance(self.transport, Attention):
                loss1 += self.transport.test(img, p1, p1_theta)
            else:
                loss1 += self.transport.test(img, p0, p1, p1_theta)
        loss0 /= n_iter
        loss1 /= n_iter

        writer.add_scalars([
            ('test_loss/attention', loss0, self.total_steps),
            ('test_loss/transport', loss1, self.total_steps),
        ])

        print(
            f'Validation: \t Attention Loss: {loss0:.4f} \t Transport Loss: {loss1:.4f}')

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        self.attention.eval_mode()
        self.transport.eval_mode()

        # Get heightmap from RGB-D images.
        img = self.get_image(obs)

        # Attention model forward pass.
        pick_conf = self.attention.forward(img)
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_conf = self.transport.forward(img, p0_pix)
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw))
        }

        # TODO(andyzeng) cleanup goal-conditioned model.

        # Make a goal image if needed, and for consistency stack with input.
        # if self.use_goal_image:
        #   cmap_g, hmap_g = utils.get_fused_heightmap(goal, self.cam_config)
        #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
        #   input_image = np.concatenate((input_image, goal_image), axis=2)
        #   assert input_image.shape[2] == 12, input_image.shape

        # if self.use_goal_image:
        #   half = int(input_image.shape[2] / 2)
        #   input_only = input_image[:, :, :half]  # ignore goal portion
        #   pick_conf = self.attention.forward(input_only)
        # else:
        # if isinstance(self.transport, TransportGoal):
        #   half = int(input_image.shape[2] / 2)
        #   img_curr = input_image[:, :, :half]
        #   img_goal = input_image[:, :, half:]
        #   place_conf = self.transport.forward(img_curr, img_goal, p0_pix)

    def get_checkpoint_names(self, n_iter):
        attention_fname = 'attention-ckpt-%d.pth' % n_iter
        transport_fname = 'transport-ckpt-%d.pth' % n_iter

        attention_fname = os.path.join(self.models_dir, attention_fname)
        transport_fname = os.path.join(self.models_dir, transport_fname)

        return attention_fname, transport_fname

    def load(self, n_iter, verbose=False):
        """Load pre-trained models."""
        attention_fname, transport_fname = self.get_checkpoint_names(n_iter)

        self.attention.load(attention_fname, verbose)
        self.transport.load(transport_fname, verbose)
        self.total_steps = n_iter

    def save(self, verbose=False):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        attention_fname, transport_fname = self.get_checkpoint_names(
            self.total_steps)

        self.attention.save(attention_fname, verbose)
        self.transport.save(transport_fname, verbose)


# -----------------------------------------------------------------------------
# Other Transporter Variants
# -----------------------------------------------------------------------------


class OriginalTransporterAgent(TransporterAgent):

    def __init__(self, name, task, n_rotations=36, verbose=False):
        super().__init__(name, task, n_rotations)

        self.attention = Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            verbose=verbose)
        self.transport = Transport(
            in_channels=self.in_shape[2],
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            verbose=verbose)


class NoTransportTransporterAgent(TransporterAgent):

    def __init__(self, name, task, n_rotations=36, verbose=False):
        super().__init__(name, task, n_rotations)

        self.attention = Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            verbose=verbose)
        self.transport = Attention(
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            preprocess=utils.preprocess,
            verbose=verbose)


class PerPixelLossTransporterAgent(TransporterAgent):

    def __init__(self, name, task, n_rotations=36, verbose=False):
        super().__init__(name, task, n_rotations)

        self.attention = Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            verbose=verbose)
        self.transport = TransportPerPixelLoss(
            in_channels=self.in_shape[2],
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            verbose=verbose)


class GoalTransporterAgent(TransporterAgent):
    """Goal-Conditioned Transporters supporting a separate goal FCN."""

    def __init__(self, name, task, n_rotations=36, verbose=False):
        super().__init__(name, task, n_rotations)

        self.attention = Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            verbose=verbose)
        self.transport = TransportGoal(
            in_channels=self.in_shape[2],
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            verbose=verbose)


class GoalNaiveTransporterAgent(TransporterAgent):
    """Naive version which stacks current and goal images through normal Transport."""

    def __init__(self, name, task, n_rotations=36, verbose=False):
        super().__init__(name, task, n_rotations)

        # Stack the goal image for the vanilla Transport module.
        t_shape = (self.in_shape[0], self.in_shape[1],
                   int(self.in_shape[2] * 2))

        self.attention = Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            verbose=verbose)
        self.transport = Transport(
            in_channels=t_shape[2],
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            verbose=verbose,
            per_pixel_loss=False,
            use_goal_image=True)
