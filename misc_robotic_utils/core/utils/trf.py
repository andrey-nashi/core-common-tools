import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def trf_euler_to_quaternion(angle: np.ndarray) -> np.ndarray:
    ax = float(angle[0])
    ay = float(angle[1])
    az = float(angle[2])

    ax /= 2.0
    ay /= 2.0
    az /= 2.0

    ci = math.cos(ax)
    si = math.sin(ax)
    cj = math.cos(ay)
    sj = math.sin(ay)
    ck = math.cos(az)
    sk = math.sin(az)

    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q

def trf_rotmat_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    rotation = R.from_matrix(matrix)
    quaternion = rotation.as_quat()
    quaternion = np.array(quaternion)
    return quaternion

def trf_rotmat_to_euler(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def trf_quaternion_to_rotmat(q: np.ndarray):
    rotation = R.from_quat(q)
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix

def pos_ori_to_matrix(position, orientation_quaternion):
    mat = trf_quaternion_to_rotmat(orientation_quaternion)
    mat = np.vstack([mat, [0, 0, 0]])
    mat = np.hstack([mat, [[position[0]], [position[1]], [position[2]], [1]]])
    return mat

def matrix_to_pos_ori(mat):
    pos = mat[:3, 3]
    ori = trf_rotmat_to_quaternion(mat[:3, :3])
    return pos, ori