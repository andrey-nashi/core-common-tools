import numpy as np
import pybullet as p
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

    def image_to_world(self, image_x, image_y, depth, view_matrix, proj_matrix, image_width, image_height):
        # Normalize image coordinates to range [-1, 1]
        ndc_x = (2.0 * image_x / image_width) - 1.0
        ndc_y = 1.0 - (2.0 * image_y / image_height)

        # Construct homogeneous coordinates in camera space
        homogeneous_coords = np.array([ndc_x * depth, ndc_y * depth, depth, 1.0])

        # Compute inverse matrices
        inv_proj_matrix = np.linalg.inv(proj_matrix)
        inv_view_matrix = np.linalg.inv(view_matrix)

        # Transform homogeneous coordinates to world coordinates
        intermediate_coords = np.dot(inv_proj_matrix, homogeneous_coords)
        intermediate_coords = intermediate_coords / intermediate_coords[3]
        world_coords = np.dot(inv_view_matrix, intermediate_coords)

        return world_coords[:3]

    def transform(self, x, y, depth):
        out = self.image_to_world(
            x, y, depth, np.array(self._view_matrix).reshape((4,4)), np.array(self._projection_matrix).reshape((4,4)), self._cam_width, self._cam_height
        )
        print(out)
        return out.tolist()
