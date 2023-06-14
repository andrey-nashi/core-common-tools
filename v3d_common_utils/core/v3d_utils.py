import numpy as np

def v3d_check_point_on_plane(plane_points: list, target_point: list):
    """
    Check if the target point is in the same plane as other 3 points
    :param plane_points: list of 3 points [[x1,y1,z1], ... [x3,y3,z3]]
    :param target_point: list of point coordinates [x,y,z]
    :return: True if 4 points are on the same plane False otherwise
    """
    vector1 = np.array(plane_points[0]) - np.array(target_point)
    vector2 = np.array(plane_points[1]) - np.array(target_point)
    vector3 = np.array(plane_points[2]) - np.array(target_point)

    cross_product = np.cross(vector1, vector2)
    dot_product = np.dot(cross_product, vector3)

    if np.isclose(dot_product, 0):
        return True
    return False