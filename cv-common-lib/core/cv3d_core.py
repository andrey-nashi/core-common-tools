import os
import cv2
import numpy as np
from scipy.ndimage import label
from .cv3d_volume import Volume


def cv3d_binarize(v: Volume, threshold, max_value: int = 255, min_value: int = 0):
    output = Volume()
    output.init_zeros_like(v)
    output.array[v.array >= threshold] = max_value
    output.array[v.array < threshold] = min_value
    return output

def cv3d_connected_components(v: Volume):
    vx = cv3d_binarize(v, 128)
    kernel = np.ones((3,3,3))
    output, l = label(vx.array, structure=kernel)
    vx.array = np.copy(output)
    return vx
