# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Mixture Density Networks utilities."""

import matplotlib.pyplot as plt
import numpy as np

import torch
from einops.layers.torch import Rearrange


EPS = 1e-12


def pick_max_mean(pi, mu, var):
    """Prediction as the mean of the most-weighted gaussian.

    Args:
      pi: (batch_size, num_gaussians)
      mu: (batch_size, num_gaussians * d_out)
      var: (batch_size, num_gaussians)

    Returns:
      (batch_size, d_out) NUMPY
    """
    del var
    batch_size, num_gaussians = pi.shape
    d_out = mu.shape[-1] // num_gaussians
    mu = Rearrange('b (n d) -> b n d', n=num_gaussians, d=d_out)(mu)

    prediction = np.zeros((batch_size, d_out))
    argmax_pi = torch.argmax(pi, dim=1)  # shape (batch_size)
    for i in range(batch_size):
        ith_argmax_pi = argmax_pi[i].numpy()
        prediction[i] = mu[i, ith_argmax_pi]
    return prediction


def sample_from_pdf(pi, mu, var, num_samples=1):
    """Prediction as a sample from the gaussian mixture.

    Args:
      pi: (batch_size, num_gaussians)
      mu: (batch_size, num_gaussians * d_out)
      var: (batch_size, num_gaussians)
      num_samples: Number of samples to draw from the pdf.
    Returns:
      (batch_size, num_samples, d_out) NUMPY
    """
    pi, mu, var = pi.numpy(), mu.numpy(), var.numpy()
    # apply temperature?
    # pi = pi**4 # apply temp
    var = var**4

    # CHECK LINE BELOW
    pi = pi * (1 / pi.sum(dim=1))

    batch_size, num_gaussians = pi.shape
    d_out = mu.shape[-1] // num_gaussians
    mu = Rearrange('b (n d) -> b n d', n=num_gaussians, d=d_out)(mu)

    samples = np.zeros((batch_size, num_samples, d_out))
    for i in range(batch_size):
        for j in range(num_samples):
            idx = np.random.choice(range(k), p=pi[i])
            draw = np.random.normal(mu[i, idx], np.sqrt(var[i, idx]))
            samples[i, j] = draw
    return samples


def multivar_gaussian_pdf(y, mu, var):
    r"""Assumes covariance matrix is identity times variance.

    i.e.
    \Sigma = I \sigma^2
    for \Sigma covariance matrix, \sigma std. deviation.

    Args:
      y: shape (batch_size, d)
      mu: shape (batch_size, k, d)
      var: shape (batch_size, k)

    Returns:
      float pdf value.
    """
    # assert len(y.shape) == 2
    # assert len(mu.shape) == 3
    # assert len(var.shape) == 2
    # assert y.shape[-1] == mu.shape[-1]
    # assert mu.shape[1] == var.shape[-1]
    # assert y.shape[0] == mu.shape[0]
    # assert y.shape[0] == var.shape[0]
    d = mu.shape[-1]
    y = y[:, np.newaxis, :]

    # CHECK FOR NANS
    dot_prod = torch.sum((y - mu)**2, dim=2)  # shape (batch_size, k)
    exp_factor = torch.divide(-1., (2. * (var))) * dot_prod
    numerator = torch.exp(exp_factor)  # shape (batch_size, k)
    denominator = torch.sqrt((2 * np.pi * (var))**d)
    return torch.multiply(numerator, 1 / denominator)


def mdn_loss(y, mdn_predictions):
    """Mixture Density Network loss.

    Args:
      y: true "y", shape (batch_size, d_out)
      mdn_predictions: tuple of:
        - pi: (batch_size, num_gaussians)
        - mu: (batch_size, num_gaussians * d_out)
        - var: (batch_size, num_gaussians)

    Returns:
      loss, scalar
    """
    pi, mu, var = mdn_predictions
    _, num_gaussians = pi.shape
    d_out = mu.shape[-1] // num_gaussians
    mu = Rearrange('b (n d) -> b n d', n=num_gaussians, d=d_out)(mu)

    # mu now (batch_size, num_gaussians, d_out) shape
    pdf = multivar_gaussian_pdf(y, mu, var)
    # multiply with each pi and sum it
    p = torch.multiply(torch.clip(pdf, 1e-8, 1e8), torch.clip(pi, 1e-8, 1e8))
    p = torch.sum(p, dim=1, keepdim=True)
    p = - torch.log(torch.clip(p, 1e-8, 1e8))
    # plot_mdn_predictions(y, mdn_predictions)
    return torch.mean(p)


def plot_mdn_predictions(y, mdn_predictions):
    """Plot Mixture Density Network Predictions.

    Args:
      y: true "y", shape (batch_size, d_out)
      mdn_predictions: tuple of:
        - pi: (batch_size, num_gaussians)
        - mu: (batch_size, num_gaussians * d_out)
        - var: (batch_size, num_gaussians)
    """
    _, ax = plt.subplots(1, 1)

    pi, mu, var = mdn_predictions

    n = 5
    y = y[:n, :]
    pi = pi[:n, :]
    mu = mu[:n, :]
    var = var[:n, :]

    ax.cla()
    ax.scatter(y[:, 0], y[:, 1])

    batch_size, d_out = y.shape
    num_gaussians = pi.shape[-1]
    mu = Rearrange('b (n d) -> (b n) d', b=batch_size,
                   n=num_gaussians, d=d_out)(mu)
    pi = Rearrange('b n -> (b n)')(pi)
    pi = torch.clip(pi, 0.01, 1.0)

    mu = mu.numpy()
    pi = pi.numpy()

    rgba_colors = np.zeros((len(pi), 4))
    # for red the first column needs to be one
    rgba_colors[:, 0] = 1.0
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = pi

    ax.scatter(mu[:, 0], mu[:, 1], color=rgba_colors)

    plt.draw()
    plt.pause(0.001)
