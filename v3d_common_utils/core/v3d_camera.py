import numpy as np
class CameraParameters:

    def __init__(self):
        self.name = None
        self.serial_number = None

        # ---- Focal
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # ---- Distortion
        self.k1 = None
        self.k2 = None
        self.k3 = None
        self.p1 = None
        self.p2 = None

        # ---- Pose matrix
        self.pose = None


    def get_matrix(self, is_numpy=True):
        """
        Intrinsic camera matrix
        :return: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        """
        matrix = [
                self.fx, 0, self.cx,
                0, self.fy, self.cy,
                0, 0, 1
            ]
        if not is_numpy:
            return matrix
        else:
            return np.asarray(matrix).reshape(3, 3).tolist()

    def get_distortion(self):
        return [self.k1, self.k2, self.k3, self.p1, self.p2]

    def get_pose(self, is_numpy=True):
        if not is_numpy:
            return self.pose
        else:
            return np.asarray(self.pose).reshape(4, 4)

    def set_matrix(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def set_distortion(self, k1: float, k2: float, k3: float, p1: float, p2: float):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2

    def set_pose(self, pose: list):
        self.pose = pose

    def convert_uvd2xyz(self, u: int, v: int, d: int = None, d_scale = 1):
        """
        Convert image coordinates to PCD coordinates
        :param u: image coordinate X
        :param v: image coordinate Y
        :param d: depth value from the depth map
        :return:
        """
        z = d / d_scale if not d is None else 1
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return [x, y, z]