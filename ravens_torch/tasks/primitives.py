# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Motion primitives."""

import numpy as np
from ravens_torch.utils import utils


class PickPlace():
    """Pick and place primitive."""

    def __init__(self, height=0.32, speed=0.01, video_recorder=None):
        self.height, self.speed = height, speed
        self.video_recorder = video_recorder
        self.mode = None

    def _set_video_recorder(self, video_recorder):
        self.video_recorder = video_recorder

    def _set_mode(self, mode):
        self.mode = mode

    def multiply(self, pose0, pose1):
        if self.mode == 'test':
            self.video_recorder.record_frame()
        return utils.multiply(pose0, pose1)

    def __call__(self, movej, movep, ee, pose0, pose1):
        """Execute pick and place primitive.

        Args:
          movej: function to move robot joints.
          movep: function to move robot end effector pose.
          ee: robot end effector.
          pose0: SE(3) picking pose.
          pose1: SE(3) placing pose.

        Returns:
          timeout: robot movement timed out if True.
        """
        pick_pose, place_pose = pose0, pose1

        # Execute picking primitive.
        prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
        postpick_to_pick = ((0, 0, self.height), (0, 0, 0, 1))
        prepick_pose = self.multiply(pick_pose, prepick_to_pick)
        postpick_pose = self.multiply(pick_pose, postpick_to_pick)
        timeout = movep(prepick_pose)

        # Move towards pick pose until contact is detected.
        delta = (np.float32([0, 0, -0.001]),
                 utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
        targ_pose = prepick_pose
        while not ee.detect_contact():  # and target_pose[2] > 0:
            targ_pose = self.multiply(targ_pose, delta)
            timeout |= movep(targ_pose)
            if timeout:
                return True

        # Activate end effector, move up, and check picking success.
        ee.activate()
        timeout |= movep(postpick_pose, self.speed)
        pick_success = ee.check_grasp()

        # Execute placing primitive if pick is successful.
        if pick_success:
            preplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
            postplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
            preplace_pose = self.multiply(place_pose, preplace_to_place)
            postplace_pose = self.multiply(place_pose, postplace_to_place)
            targ_pose = preplace_pose
            while not ee.detect_contact():
                targ_pose = self.multiply(targ_pose, delta)
                timeout |= movep(targ_pose, self.speed)
                if timeout:
                    return True
            ee.release()
            timeout |= movep(postplace_pose)

        # Move to prepick pose if pick is not successful.
        else:
            ee.release()
            timeout |= movep(prepick_pose)

        return timeout


class Push():
    """Pushing with a brush primitive."""

    def __init__(self, video_recorder=None):
        self.video_recorder = video_recorder
        self.mode = None

    def _set_video_recorder(self, video_recorder):
        self.video_recorder = video_recorder

    def _set_mode(self, mode):
        self.mode = mode

    def __call__(self, movej, movep, ee, pose0, pose1):
        """Execute pushing primitive.

        Args:
        movej: function to move robot joints.
        movep: function to move robot end effector pose.
        ee: robot end effector.
        pose0: SE(3) starting pose.
        pose1: SE(3) ending pose.

        Returns:
        timeout: robot movement timed out if True.
        """

        # Adjust push start and end positions.
        pos0 = np.float32((pose0[0][0], pose0[0][1], 0.005))
        pos1 = np.float32((pose1[0][0], pose1[0][1], 0.005))
        vec = np.float32(pos1) - np.float32(pos0)
        length = np.linalg.norm(vec)
        vec = vec / length
        pos0 -= vec * 0.02
        pos1 -= vec * 0.05

        # Align spatula against push direction.
        theta = np.arctan2(vec[1], vec[0])
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

        over0 = (pos0[0], pos0[1], 0.31)
        over1 = (pos1[0], pos1[1], 0.31)

        # Execute push.
        timeout = movep((over0, rot))
        timeout |= movep((pos0, rot))
        n_push = np.int32(np.floor(np.linalg.norm(pos1 - pos0) / 0.01))
        for _ in range(n_push):
            target = pos0 + vec * n_push * 0.01
            timeout |= movep((target, rot), speed=0.003)
        timeout |= movep((pos1, rot), speed=0.003)
        timeout |= movep((over1, rot))

        return timeout
