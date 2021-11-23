# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Image utilities."""

import cv2
import numpy as np
from transforms3d import euler
from PIL import Image
import torchvision
from torchvision.transforms.functional import rotate
from einops.layers.torch import Rearrange

from ravens_torch.utils.utils.heightmap import reconstruct_heightmaps, pix_to_xyz


# -----------------------------------------------------------------------------
# IMAGE UTILS
# -----------------------------------------------------------------------------


def preprocess(img):
    """Pre-process input (subtract mean, divide by std)."""
    color_mean = 0.18877631
    depth_mean = 0.00509261
    color_std = 0.07276466
    depth_std = 0.00903967
    img[:, :, :3] = (img[:, :, :3] / 255 - color_mean) / color_std
    img[:, :, 3:] = (img[:, :, 3:] - depth_mean) / depth_std
    return img


def get_fused_heightmap(obs, configs, bounds, pix_size):
    """Reconstruct orthographic heightmaps with segmentation masks."""
    heightmaps, colormaps = reconstruct_heightmaps(
        obs['color'], obs['depth'], configs, bounds, pix_size)
    colormaps = np.float32(colormaps)
    heightmaps = np.float32(heightmaps)

    # Fuse maps from different views.
    valid = np.sum(colormaps, axis=3) > 0
    repeat = np.sum(valid, axis=0)
    repeat[repeat == 0] = 1
    cmap = np.sum(colormaps, axis=0) / repeat[Ellipsis, None]
    cmap = np.uint8(np.round(cmap))
    hmap = np.max(heightmaps, axis=0)  # Max to handle occlusions.
    return cmap, hmap


def get_image_transform(theta, trans, pivot=(0, 0)):
    """Compute composite 2D rigid transformation matrix."""
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_t_image = np.array([[1., 0., -pivot[0]], [0., 1., -pivot[1]],
                              [0., 0., 1.]])
    image_t_pivot = np.array([[1., 0., pivot[0]], [0., 1., pivot[1]],
                              [0., 0., 1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]], [0., 0., 1.]])
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))


def check_transform(image, pixel, transform):
    """Valid transform only if pixel locations are still in FoV after transform."""
    new_pixel = np.flip(
        np.int32(
            np.round(
                np.dot(transform,
                       np.float32([pixel[1], pixel[0],
                                   1.]).reshape(3, 1))))[:2].squeeze())
    valid = np.all(
        new_pixel >= 0
    ) and new_pixel[0] < image.shape[0] and new_pixel[1] < image.shape[1]
    return valid, new_pixel


def get_se3_from_image_transform(theta, trans, pivot, heightmap, bounds,
                                 pixel_size):
    """Calculate SE3 from image transform."""
    position_center = pix_to_xyz(
        np.flip(np.int32(np.round(pivot))),
        heightmap,
        bounds,
        pixel_size,
        skip_height=False)
    new_position_center = pix_to_xyz(
        np.flip(np.int32(np.round(pivot + trans))),
        heightmap,
        bounds,
        pixel_size,
        skip_height=True)
    # Don't look up the z height, it might get augmented out of frame
    new_position_center = (new_position_center[0], new_position_center[1],
                           position_center[2])

    delta_position = np.array(new_position_center) - np.array(position_center)

    t_world_center = np.eye(4)
    t_world_center[0:3, 3] = np.array(position_center)

    t_centernew_center = np.eye(4)
    euler_zxy = (-theta, 0, 0)
    t_centernew_center[0:3, 0:3] = euler.euler2mat(
        *euler_zxy, axes='szxy')[0:3, 0:3]

    t_centernew_center_tonly = np.eye(4)
    t_centernew_center_tonly[0:3, 3] = -delta_position
    t_centernew_center = t_centernew_center @ t_centernew_center_tonly

    t_world_centernew = t_world_center @ np.linalg.inv(t_centernew_center)
    return t_world_center, t_world_centernew


def get_random_image_transform_params(image_size):
    theta_sigma = 2 * np.pi / 6
    theta = np.random.normal(0, theta_sigma)

    trans_sigma = np.min(image_size) / 6
    trans = np.random.normal(0, trans_sigma, size=2)  # [x, y]
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot


def perturb(input_image, pixels, set_theta_zero=False):
    """Data augmentation on images."""
    image_size = input_image.shape[:2]

    # Compute random rigid transform.
    while True:
        theta, trans, pivot = get_random_image_transform_params(image_size)
        if set_theta_zero:
            theta = 0.
        transform = get_image_transform(theta, trans, pivot)
        transform_params = theta, trans, pivot

        # Ensure pixels remain in the image after transform.
        is_valid = True
        new_pixels = []
        new_rounded_pixels = []
        for pixel in pixels:
            pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)

            rounded_pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
            rounded_pixel = np.flip(rounded_pixel)

            pixel = (transform @ pixel)[:2].squeeze()
            pixel = np.flip(pixel)

            in_fov_rounded = rounded_pixel[0] < image_size[0] and rounded_pixel[
                1] < image_size[1]
            in_fov = pixel[0] < image_size[0] and pixel[1] < image_size[1]

            is_valid = is_valid and np.all(rounded_pixel >= 0) and np.all(
                pixel >= 0) and in_fov_rounded and in_fov

            new_pixels.append(pixel)
            new_rounded_pixels.append(rounded_pixel)
        if is_valid:
            break

    # Apply rigid transform to image and pixel labels.
    input_image = cv2.warpAffine(
        input_image,
        transform[:2, :], (image_size[1], image_size[0]),
        flags=cv2.INTER_NEAREST)
    return input_image, new_pixels, new_rounded_pixels, transform_params


def apply_rotations_to_tensor(in_tensor, num_rotations, center=None, reverse=False):
    if reverse:
        thetas = [-i * 360 / num_rotations for i in range(num_rotations)]
    else:
        thetas = [i * 360 / num_rotations for i in range(num_rotations)]
    thetas = np.array(thetas)

    tensor = in_tensor.clone()
    if tensor.shape[0] == 1:    # (1,h,w,c)
        tensor = tensor.repeat(
            (num_rotations, 1, 1, 1))  # (num_rotations,h,w,c)

    tensor = Rearrange('b h w c -> b c h w')(tensor)
    t_clone = tensor.clone()
    for idx, theta in enumerate(thetas):
        tensor[idx, ...] = rotate(
            t_clone[idx, ...],
            theta,
            center=center,
            resample=Image.NEAREST)
    tensor = Rearrange('b c h w -> b h w c')(tensor)

    return tensor
