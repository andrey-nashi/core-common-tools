import numpy as np
class CameraParameters:

    def __init__(self, name: str = None, serial_number: str = None, fx: float = None, fy: float = None, cx: float = None, cy: float = None,
                 k1: float = None, k2: float = None, k3: float = None, p1: float = None, p2: float = None, pose: list = None):
        self.name = name
        self.serial_number = serial_number

        # ---- Focal
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # ---- Distortion
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2

        # ---- Pose matrix
        self.pose = pose
    @property
    def distortion(self, is_numpy=True):
        distortion = [self.k1, self.k2, self.k3, self.p1, self.p2]
        if not is_numpy: return distortion
        else: return np.asarray(distortion)


    @property
    def matrix_intrinsic(self, is_numpy=True):
        """
        Intrinsic camera matrix
        :return: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        """
        matrix = [
                self.fx, 0, self.cx,
                0, self.fy, self.cy,
                0, 0, 1
            ]
        if not is_numpy: return matrix
        else: return np.asarray(matrix).reshape(3, 3)

    @property
    def matrix_extrinsic(self, is_numpy=True):
        if not is_numpy: return self.pose
        else: return np.asarray(self.pose).reshape(4, 4)



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

    def convert_uvd2xyz(self, u: int, v: int, d: int = None, d_scale: float = 1):
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