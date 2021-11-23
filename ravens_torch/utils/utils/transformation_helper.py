# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Transformation Helper utilities."""

import numpy as np
from transforms3d import euler
import pybullet as p


# -------------------------------------------------------------------------
# Transformation Helper Functions
# -------------------------------------------------------------------------


def invert(pose):
    return p.invertTransform(pose[0], pose[1])


def multiply(pose0, pose1):
    return p.multiplyTransforms(pose0[0], pose0[1], pose1[0], pose1[1])


def apply(pose, position):
    position = np.float32(position)
    position_shape = position.shape
    position = np.float32(position).reshape(3, -1)
    rotation = np.float32(p.getMatrixFromQuaternion(pose[1])).reshape(3, 3)
    translation = np.float32(pose[0]).reshape(3, 1)
    position = rotation @ position + translation
    return tuple(position.reshape(position_shape))


def eulerXYZ_to_quatXYZW(rotation):  # pylint: disable=invalid-name
    """Abstraction for converting from a 3-parameter rotation to quaterion.

    This will help us easily switch which rotation parameterization we use.
    Quaternion should be in xyzw order for pybullet.

    Args:
      rotation: a 3-parameter rotation, in xyz order tuple of 3 floats

    Returns:
      quaternion, in xyzw order, tuple of 4 floats
    """
    euler_zxy = (rotation[2], rotation[0], rotation[1])
    quaternion_wxyz = euler.euler2quat(*euler_zxy, axes='szxy')
    q = quaternion_wxyz
    quaternion_xyzw = (q[1], q[2], q[3], q[0])
    return quaternion_xyzw


def quatXYZW_to_eulerXYZ(quaternion_xyzw):  # pylint: disable=invalid-name
    """Abstraction for converting from quaternion to a 3-parameter toation.

    This will help us easily switch which rotation parameterization we use.
    Quaternion should be in xyzw order for pybullet.

    Args:
      quaternion_xyzw: in xyzw order, tuple of 4 floats

    Returns:
      rotation: a 3-parameter rotation, in xyz order, tuple of 3 floats
    """
    q = quaternion_xyzw
    quaternion_wxyz = np.array([q[3], q[0], q[1], q[2]])
    euler_zxy = euler.quat2euler(quaternion_wxyz, axes='szxy')
    euler_xyz = (euler_zxy[1], euler_zxy[2], euler_zxy[0])
    return euler_xyz


def apply_transform(transform_to_from, points_from):
    r"""Transforms points (3D) into new frame.

    Using transform_to_from notation.

    Args:
      transform_to_from: numpy.ndarray of shape [B,4,4], SE3
      points_from: numpy.ndarray of shape [B,3,N]

    Returns:
      points_to: numpy.ndarray of shape [B,3,N]
    """
    num_points = points_from.shape[-1]

    # non-batched
    if len(transform_to_from.shape) == 2:
        ones = np.ones((1, num_points))

        # makes these each into homogenous vectors
        points_from = np.vstack((points_from, ones))  # [4,N]
        points_to = transform_to_from @ points_from  # [4,N]
        return points_to[0:3, :]  # [3,N]

    # batched
    else:
        assert len(transform_to_from.shape) == 3
        batch_size = transform_to_from.shape[0]
        zeros = np.ones((batch_size, 1, num_points))
        points_from = np.concatenate((points_from, zeros), axis=1)
        assert points_from.shape[1] == 4
        points_to = transform_to_from @ points_from
        return points_to[:, 0:3, :]
