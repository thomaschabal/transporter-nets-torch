import os
import numpy as np
import skvideo.io
import pybullet as p

from ravens_torch.tasks import cameras
from ravens_torch.utils.text import bold

CONFIG = cameras.RealSenseD415.CONFIG


class CameraImageGetter:
    def __init__(self, camera_config=None, width=1280):

        self.camera_idx = 0
        if camera_config is not None:
            self.config = camera_config
        else:
            self.config = CONFIG[self.camera_idx]
        self.width = width

        self._compute_view_matrix()
        self._compute_projection_matrix_fov()

    def _compute_view_matrix(self):
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(self.config['rotation'])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)

        camera_position = self.config['position']
        camera_target_position = self.config['position'] + lookdir
        camera_up_vector = (rotm @ updir).reshape(-1)

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=camera_target_position,
            cameraUpVector=camera_up_vector)

    def _compute_projection_matrix_fov(self):
        im_w, im_h = self.config['image_size']
        focal_len = self.config['intrinsics'][0]
        znear, zfar = self.config['zrange']
        fovh = (im_w / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi
        aspect_ratio = im_h / im_w

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=fovh,
            aspect=aspect_ratio,
            nearVal=znear,
            farVal=zfar)

    def __call__(self):
        height = int(self.width * 3 / 4)
        _, _, rgbImg, _, _ = p.getCameraImage(
            width=self.width,
            height=height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix)

        return rgbImg


class VideoRecorder:
    """
        Video Recorder for the PyBullet environment
        Call VideoRecorder with a wrapper:
            `with VideoRecorder(...) as vid_rec:`
        Call `record_frame` wherever you need to save a frame in your code
            (for instace: after `p.stepSimulation()`, after `p.multiplyTransforms`...).
    """

    def __init__(self, save_dir, episode_idx=0, record_mp4=True, display=True, verbose=False, camera_config=None):
        self.record_mp4 = record_mp4
        self.display = display
        self.verbose = verbose
        self.record_every = 5
        self.n_steps = 0

        self.video_name = f"{save_dir}/episode-{episode_idx}.mp4"
        if record_mp4:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if verbose and record_mp4:
            print(
                f"{bold('Video Recorder')} active, will save videos at {bold(self.video_name)}")

        width = 1280
        self.camera_image_getter = CameraImageGetter(
            camera_config=camera_config, width=width)

    def __enter__(self):
        if self.record_mp4:
            self.frames = []
        return self

    def record_frame(self):
        if self.record_mp4 and self.n_steps % self.record_every == 0:
            rgbImg = self.camera_image_getter()[..., :3]
            self.frames.append(rgbImg)
        self.n_steps += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.record_mp4:
            if self.verbose:
                print(f"Saving video at {self.video_name}")

            skvideo.io.vwrite(
                self.video_name,
                np.array(self.frames),
                inputdict={'-r': "120/1"})
