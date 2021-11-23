# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Meshcat utilities."""

import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf

from ravens_torch.utils.utils.heightmap import unproject_depth_vectorized
from ravens_torch.utils.utils.transformation_helper import apply_transform


# -----------------------------------------------------------------------------
# MESHCAT UTILS
# -----------------------------------------------------------------------------


def create_visualizer(clear=True):
    print('Waiting for meshcat server... have you started a server?')
    vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    if clear:
        vis.delete()
    return vis


def make_frame(vis, name, h, radius, o=1.0):
    """Add a red-green-blue triad to the Meschat visualizer.

    Args:
      vis (MeshCat Visualizer): the visualizer
      name (string): name for this frame (should be unique)
      h (float): height of frame visualization
      radius (float): radius of frame visualization
      o (float): opacity
    """
    vis[name]['x'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0xff0000, reflectivity=0.8, opacity=o))
    rotate_x = mtf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
    rotate_x[0, 3] = h / 2
    vis[name]['x'].set_transform(rotate_x)

    vis[name]['y'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x00ff00, reflectivity=0.8, opacity=o))
    rotate_y = mtf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    rotate_y[1, 3] = h / 2
    vis[name]['y'].set_transform(rotate_y)

    vis[name]['z'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x0000ff, reflectivity=0.8, opacity=o))
    rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    rotate_z[2, 3] = h / 2
    vis[name]['z'].set_transform(rotate_z)


def meshcat_visualize(vis, obs, act, info):
    """Visualize data using meshcat."""

    for key in sorted(info.keys()):

        pose = info[key]
        pick_transform = np.eye(4)
        pick_transform[0:3, 3] = pose[0]
        quaternion_wxyz = np.asarray(
            [pose[1][3], pose[1][0], pose[1][1], pose[1][2]])
        pick_transform[0:3, 0:3] = mtf.quaternion_matrix(quaternion_wxyz)[
            0:3, 0:3]
        label = 'obj_' + str(key)
        make_frame(vis, label, h=0.05, radius=0.0012, o=1.0)
        vis[label].set_transform(pick_transform)

    for cam_index in range(len(act['camera_config'])):

        verts = unproject_depth_vectorized(
            obs['depth'][cam_index], np.array([0, 1]),
            np.array(act['camera_config'][cam_index]
                     ['intrinsics']).reshape(3, 3),
            np.zeros(5))

        # switch from [N,3] to [3,N]
        verts = verts.T

        cam_transform = np.eye(4)
        cam_transform[0:3, 3] = act['camera_config'][cam_index]['position']
        quaternion_xyzw = act['camera_config'][cam_index]['rotation']
        quaternion_wxyz = np.asarray([
            quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1],
            quaternion_xyzw[2]
        ])
        cam_transform[0:3, 0:3] = mtf.quaternion_matrix(quaternion_wxyz)[
            0:3, 0:3]
        verts = apply_transform(cam_transform, verts)

        colors = obs['color'][cam_index].reshape(-1, 3).T / 255.0

        vis['pointclouds/' + str(cam_index)].set_object(
            g.PointCloud(position=verts, color=colors))
