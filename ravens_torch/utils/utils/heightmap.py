# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Heightmap utilities."""

import cv2
import numpy as np
import pybullet as p

# -----------------------------------------------------------------------------
# HEIGHTMAP UTILS
# -----------------------------------------------------------------------------


def get_heightmap(points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in world coordinates.
      colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
      bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
        region in 3D space to generate heightmap in world coordinates.
      pixel_size: float defining size of each pixel in meters.

    Returns:
      heightmap: HxW float array of height (from lower z-bound) in meters.
      colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
    """
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (
        points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (
        points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (
        points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
    return heightmap, colormap


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.

    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in camera coordinates.
      transform: 4x4 float array representing a rigid transformation matrix.

    Returns:
      points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                            'constant', constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points


def reconstruct_heightmaps(color, depth, configs, bounds, pixel_size):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
    heightmaps, colormaps = [], []
    for color, depth, config in zip(color, depth, configs):
        intrinsics = np.array(config['intrinsics']).reshape(3, 3)
        xyz = get_pointcloud(depth, intrinsics)
        position = np.array(config['position']).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        xyz = transform_pointcloud(xyz, transform)
        heightmap, colormap = get_heightmap(xyz, color, bounds, pixel_size)
        heightmaps.append(heightmap)
        colormaps.append(colormap)
    return heightmaps, colormaps


def pix_to_xyz(pixel, height, bounds, pixel_size, skip_height=False):
    """Convert from pixel location on heightmap to 3D position."""
    u, v = pixel
    x = bounds[0, 0] + v * pixel_size
    y = bounds[1, 0] + u * pixel_size
    if not skip_height:
        z = bounds[2, 0] + height[u, v]
    else:
        z = 0.0
    return (x, y, z)


def xyz_to_pix(position, bounds, pixel_size):
    """Convert from 3D position to pixel location on heightmap."""
    u = int(np.round((position[1] - bounds[1, 0]) / pixel_size))
    v = int(np.round((position[0] - bounds[0, 0]) / pixel_size))
    return (u, v)


def unproject_vectorized(uv_coordinates, depth_values,
                         intrinsic,
                         distortion):
    """Vectorized version of unproject(), for N points.

    Args:
      uv_coordinates: pixel coordinates to unproject of shape (n, 2).
      depth_values: depth values corresponding index-wise to the uv_coordinates of
        shape (n).
      intrinsic: array of shape (3, 3). This is typically the return value of
        intrinsics_to_matrix.
      distortion: camera distortion parameters of shape (5,).

    Returns:
      xyz coordinates in camera frame of shape (n, 3).
    """
    cam_mtx = intrinsic  # shape [3, 3]
    cam_dist = np.array(distortion)  # shape [5]

    # shape of points_undistorted is [N, 2] after the squeeze().
    points_undistorted = cv2.undistortPoints(
        uv_coordinates.reshape((-1, 1, 2)), cam_mtx, cam_dist).squeeze()

    x = points_undistorted[:, 0] * depth_values
    y = points_undistorted[:, 1] * depth_values

    xyz = np.vstack((x, y, depth_values)).T
    return xyz


def unproject_depth_vectorized(im_depth, depth_dist,
                               camera_mtx,
                               camera_dist):
    """Unproject depth image into 3D point cloud, using calibration.

    Args:
      im_depth: raw depth image, pre-calibration of shape (height, width).
      depth_dist: depth distortion parameters of shape (8,)
      camera_mtx: intrinsics matrix of shape (3, 3). This is typically the return
        value of intrinsics_to_matrix.
      camera_dist: camera distortion parameters shape (5,).

    Returns:
      numpy array of shape [3, H*W]. each column is xyz coordinates
    """
    h, w = im_depth.shape

    # shape of each u_map, v_map is [H, W].
    u_map, v_map = np.meshgrid(np.linspace(
        0, w - 1, w), np.linspace(0, h - 1, h))

    adjusted_depth = depth_dist[0] + im_depth * depth_dist[1]

    # shape after stack is [N, 2], where N = H * W.
    uv_coordinates = np.stack((u_map.reshape(-1), v_map.reshape(-1)), axis=-1)

    return unproject_vectorized(uv_coordinates, adjusted_depth.reshape(-1),
                                camera_mtx, camera_dist)
