# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""MLP ground-truth state module."""

import torch
import torch.nn as nn


ACTIVATIONS = {
    'relu': nn.ReLU,
}


def init_normal_weights_bias(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight)
        torch.nn.init.normal_(m.bias)


def DenseBlock(in_channels, out_channels, activation=None):
    if activation is not None:
        fc = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            activation(),
        )
    else:
        fc = nn.Linear(in_channels, out_channels)

    fc.apply(init_normal_weights_bias)

    return fc


class MlpModel(nn.Module):
    """MLP ground-truth state module."""

    def __init__(self,
                 d_obs,
                 d_action,
                 activation="relu",
                 mdn=False,
                 dropout=0.2,
                 use_sinusoid=True):
        super(MlpModel, self).__init__()
        self.normalize_input = True

        self.use_sinusoid = use_sinusoid
        if self.use_sinusoid:
            k = 3
        else:
            k = 1

        activation = ACTIVATIONS[activation]

        dim_concat = 128 + d_obs * k

        # CHECK INPUT DIMENSION
        self.fc1 = DenseBlock(d_obs * k, 128, activation)
        self.drop1 = nn.Dropout(p=dropout)

        self.fc2 = DenseBlock(dim_concat, 128, activation)
        self.drop2 = nn.Dropout(p=dropout)

        self.fc3 = DenseBlock(dim_concat, d_action, activation)

        self.mdn = mdn
        if self.mdn:
            k = 26
            self.mu = DenseBlock(dim_concat, d_action * k)

            # Variance should be non-negative, so exp()
            self.logvar = DenseBlock(dim_concat, k)

            # mixing coefficient should sum to 1.0, so apply softmax
            self.pi = DenseBlock(dim_concat, k)

            self.softmax = nn.Softmax()
            self.temperature = 2.5

    def reset_states(self):
        pass

    def set_normalization_parameters(self, obs_train_parameters):
        """Set normalization parameters.

        Args:
          obs_train_parameters: dict with key, values:
            - 'mean', numpy.ndarray of shape (obs_dimension)
            - 'std', numpy.ndarray of shape (obs_dimension)
        """
        self.obs_train_mean = obs_train_parameters["mean"]
        self.obs_train_std = obs_train_parameters["std"]

    def call(self, x):
        """FPROP through module.

        Args:
          x: shape: (batch_size, obs_dimension)

        Returns:
          shape: (batch_size, action_dimension)  (if MDN)
          shape of pi: (batch_size, num_gaussians)
          shape of mu: (batch_size, num_gaussians*action_dimension)
          shape of var: (batch_size, num_gaussians)
        """
        obs = x * 1.0

        # if self.normalize_input:
        #   x = (x - self.obs_train_mean) / (self.obs_train_std + 1e-7)

        def cs(x):
            if self.use_sinusoid:
                sin = torch.sin(x)
                cos = torch.cos(x)
                return torch.cat((x, cos, sin), dim=1)
            else:
                return x

        cs_obs = cs(obs)
        x = self.drop1(self.fc1(cs_obs))
        x = self.drop2(self.fc2(torch.cat((x, cs_obs), dim=1)))

        x = torch.cat((x, cs_obs), dim=1)

        if not self.mdn:
            x = self.fc3(x)
            return x

        else:
            pi = self.pi(x)
            pi = pi / self.temperature
            pi = self.softmax(pi)

            mu = self.mu(x)
            var = torch.exp(self.logvar(x))
            return (pi, mu, var)
