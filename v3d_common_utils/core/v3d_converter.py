import blosc
import numpy as np

NAN_MODE_MIN = 0
NAN_MODE_MAX = 1
NAN_MODE_MEAN = 2
NAN_MODE_MEDIAN = 3
def convert_pcd2png(pcd: np.ndarray, nan_mode: int = NAN_MODE_MIN) -> np.ndarray:
    """
    Convert point cloud stored in a numpy file into numpy depth map image.
    The stored numpy array is [H, W, 3] where each pixel is [x,y,z] coordinates.
    :param pcd: [H,W,3] numpy array
    :param nan_mode: conversion mode for NAN values in the numpy array
    - NAN_MODE_MIN | 0 - nan=min(numpy)
    - NAN_MODE_MAX | 1 - nan=max(numpy)
    - NAN_MODE_MEAN | 2 - nan=mean(numpy)
    - NAN_MODE_MEDIAN | 3 - nan=median(numpy)
    :return: numpy array a depth map representation of size [H,W] where each pixel in [0, 255]
    """
    depth_data = np.copy(pcd[..., -1])

    if np.any(np.isnan(depth_data)):

        if nan_mode == NAN_MODE_MIN: swap_value = np.nanmin(depth_data)
        elif nan_mode == NAN_MODE_MAX: swap_value = np.nanmax(depth_data)
        elif nan_mode == NAN_MODE_MEAN: swap_value = np.nanmean(depth_data)
        elif nan_mode == NAN_MODE_MEDIAN: swap_value = np.nanmedian(depth_data)
        else: swap_value = 0

        min_value = np.nanmin(depth_data)
        max_value = np.nanmax(depth_data)
        depth_data = np.nan_to_num(depth_data, nan=swap_value)


    else:
        min_value = np.min(depth_data)
        max_value = np.max(depth_data)

    depth_data = (depth_data - min_value) / (max_value - min_value)
    depth_data = depth_data * 255
    depth_data = depth_data.astype("uint8")

    return depth_data


def convert_pcdbin2png(path_file: str, nan_mode: int = NAN_MODE_MIN) -> np.ndarray:
    """
    Convert point cloud stored in a binary file into numpy image.
    The stored numpy array is [H, W, 3] where each pixel is [x,y,z] coordinates.
    :param path_file: path to '.bin' file
    :param nan_mode: conversion mode for NAN values in the numpy array
    - NAN_MODE_MIN | 0 - nan=min(numpy)
    - NAN_MODE_MAX | 1 - nan=max(numpy)
    - NAN_MODE_MEAN | 2 - nan=mean(numpy)
    - NAN_MODE_MEDIAN | 3 - nan=median(numpy)
    :return: numpy array a depth map representation of size [H,W] where each pixel in [0, 255]
    """
    f = open(path_file, "rb")
    compressed_array = f.read()
    f.close()
    pcd = blosc.unpack_array(compressed_array)
    return convert_pcd2png(pcd, nan_mode)



