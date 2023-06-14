import numpy as np

def v3d_check_point_on_plane(plane_points: list, target_point: list):
    vector1 = plane_points[0] - target_point
    vector2 = plane_points[1] - target_point
    vector3 = plane_points[2] - target_point

    cross_product = np.cross(vector1, vector2)
    dot_product = np.dot(cross_product, vector3)

    # Check if the dot product is approximately zero
    if np.isclose(dot_product, 0):
        return True
    return False