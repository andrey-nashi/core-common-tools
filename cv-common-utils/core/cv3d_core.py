import numpy as np
from scipy.ndimage import label
from .cv3d_volume import Volume


def cv3d_binarize(v: Volume, threshold, max_value: int = 255, min_value: int = 0) -> Volume:
    """
    Binarize all slices in the given volume at the given threshold, thus creating a new
    volume where each voxel (x,y,z) is min_value if v(x,y,z) < threshold and max_value otherwise
    :param v: initial volume
    :param threshold: threshold to split each voxel into min or max values
    :param max_value: min value of the voxel in the resulting volume
    :param min_value: max value of the voxel in the resulting volume
    :return: volume with binary slices, each voxel is either min_value or max_value
    """
    output = Volume()
    output.init_zeros_like(v)
    output.array[v.array >= threshold] = max_value
    output.array[v.array < threshold] = min_value
    return output

def cv3d_connected_components(v: Volume, threshold: int = 128) -> Volume:
    """
    Execute 3D connected components for the given volume, for which binarization will be applied
    at threshold level = 128. The algorithm uses `label` function from scipy using (3,3,3) kernel.
    :param v: input volume (x,y,z)
    :return: volume, where each voxel is the ID of the connected region
    """
    vx = cv3d_binarize(v, threshold)
    kernel = np.ones((3,3,3))
    output, l = label(vx.array, structure=kernel)
    vx.array = np.copy(output)
    return vx
