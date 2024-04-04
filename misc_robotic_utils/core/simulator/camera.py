import math
import random

import numpy as np
import pybullet as p
import cv2

class Camera:

    DEFAULT_LOOK_AT = [0, 0, 0]
    DEFAULT_DISTANCE = 1
    DEFAULT_YAW = 0
    DEFAULT_PITCH = -90
    DEFAULT_ROLL = 0
    DEFAULT_AXIS_UP = 2
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_FOV = 60
    DEFAULT_PLANE_NEAR = 0.1
    DEFAULT_PLANE_FAR = 2

    def __init__(self, look_at: list = DEFAULT_LOOK_AT, distance: float = DEFAULT_DISTANCE, yaw: float = DEFAULT_YAW,
                 pitch: float = DEFAULT_PITCH, roll: float = DEFAULT_ROLL, axis_up: int = DEFAULT_AXIS_UP,
                 width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT, fov: int = DEFAULT_FOV,
                 th_plane_near: int = DEFAULT_PLANE_NEAR, th_plane_far: int = DEFAULT_PLANE_FAR):
        """
        Initialize pybullet camera, default one is oriented to look from top directly down
        :param look_at: XYZ point at which the camera is looking at
        :param distance: distance from the 'look at' point to camera
        :param yaw: yaw angle in degrees (rotation around the vertical axis)
        :param pitch: pitch angle in degrees (rotation around the horizontal axis)
        :param roll: roll angle in degrees (rotation around the line of sight)
        :param axis_up: index of the up axis (2 for z-axis)
        :param width: resolution of the camera (width)
        :param height: resolution of the camera (height)
        :param fov: field of view
        :param th_plane_near: near clipping plane
        :param th_plane_far: far clipping plane
        """


        # ---- Point XYZ where the camera looks at
        self._cam_look_at = look_at
        # ---- Distance from look at point to camera
        self._cam_distance = distance
        # ---- Orientation of the camera
        self._cam_yaw = yaw
        self._cam_pitch = pitch
        self._cam_roll = roll
        # ---- UP orientation of the camera
        self._cam_up = axis_up
        # ---- Resolution
        self._cam_width = width
        self._cam_height = height
        # ---- Field of view
        self._cam_fov = fov

        # ---- Camera thresholds by distance
        self._cam_plane_near = th_plane_near
        self._cam_plane_far = th_plane_far

        # --------------------------------
        self._view_matrix = None
        self._projection_matrix = None

        self._is_open = False
        self._buffer_rgb = None
        self._buffer_depth = None
        self._buffer_mask = None

    def open(self) -> None:
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(self._cam_look_at, self._cam_distance, self._cam_yaw,
                                                                self._cam_pitch, self._cam_roll, self._cam_up)


        self._projection_matrix = p.computeProjectionMatrixFOV(self._cam_fov, self._cam_width / self._cam_height,
                                                               self._cam_plane_near, self._cam_plane_far)

        self._is_open = True

    def capture(self) -> bool:
        if not self._is_open:
            return False

        image = p.getCameraImage(self._cam_width, self._cam_height, self._view_matrix, self._projection_matrix)

        self._buffer_rgb = image[2]
        self._buffer_depth = image[3]
        self._buffer_mask = image[4]

        return True
    def get_rgb(self):
        return self._buffer_rgb

    def get_depth(self):
        return self._buffer_depth

    def get_mask(self):
        return self._buffer_mask


    def find_object(self, obj_ids):
        seg = self.get_mask()
        depth = self.get_depth()
        rgb = self.get_rgb()
        rgb = rgb.astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        obj_ids_in_frame = np.unique(seg).tolist()
        obj_ids_in_frame = [idx for idx in obj_ids_in_frame if idx in obj_ids]
        if len(obj_ids_in_frame) == 0:
            return None
        obj_id = obj_ids_in_frame[random.randint(0, len(obj_ids_in_frame) -1)]

        y = int(np.mean(np.argwhere(seg == obj_id).T[0]))
        x = int(np.mean(np.argwhere(seg == obj_id).T[1]))
        d = depth[y, x]
        print(np.min(depth), np.max(depth))
        out = np.copy(seg) * 20
        out = out.astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        cv2.circle(out, [int(x), int(y)], 5, [255, 0, 0], -1)
        cv2.imwrite("segmentation.png", out)
        cv2.imwrite("depth.png", (depth * 255).astype(np.uint8))
        cv2.imwrite("image.png", rgb)

        d = (d - self._cam_plane_near) / (self._cam_plane_far - self._cam_plane_near)
        obj_pose = self.image_to_world_point(x, y, d, self._view_matrix, self._projection_matrix, self._cam_width, self._cam_height)

        x_n = 2 * x / self._cam_width - 1
        y_n = 2 * y / self._cam_height - 1
        z_n = (d - self._cam_plane_near) / (self._cam_plane_far - self._cam_plane_near)

        obj_pose_r = p.getBasePositionAndOrientation(obj_id)[0]
        print(obj_pose, obj_pose_r)
        return obj_id, obj_pose_r

    def image_to_world_point(self, x, y, depth, view_matrix, projection_matrix, image_width, image_height):
        # Step 1: Image Coordinates to NDC
        x_ndc = (2 * x) / image_width - 1
        y_ndc = 1 - (2 * y) / image_height  # Inverting y to match the NDC space
        z_ndc = 2 * depth - 1  # Assuming depth is normalized [0, 1]

        ndc_point = np.array([x_ndc, y_ndc, z_ndc, 1])

        # Step 2: NDC to Camera Space
        inv_projection_matrix = np.linalg.inv(np.array(projection_matrix).reshape(4, 4))
        camera_space_point = np.dot(inv_projection_matrix, ndc_point)
        camera_space_point /= camera_space_point[3]  # Dehomogenize

        print(camera_space_point)
        # Step 3: Camera Space to World Space
        inv_view_matrix = np.linalg.inv(np.array(view_matrix).reshape(4, 4))
        world_space_point = np.dot(inv_view_matrix, camera_space_point)
        world_space_point /= world_space_point[3]  # Dehomogenize

        return world_space_point[:3]  # Returning XYZ, omitting the w component
