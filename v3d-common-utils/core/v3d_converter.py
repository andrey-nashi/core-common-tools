import blosc
import numpy as np


NAN_MODE_MIN = 0
NAN_MODE_MAX = 1
NAN_MODE_MEAN = 2
NAN_MODE_MEDIAN = 3
def convert_pcd2png(pcd: np.ndarray, nan_mode: int = NAN_MODE_MIN) -> np.ndarray:
    depth_data = pcd[..., -1]

    if np.any(np.isnan(depth_data)):
        if nan_mode == NAN_MODE_MIN: swap_value = np.nanmin(depth_data)
        elif nan_mode == NAN_MODE_MAX: swap_value = np.nanmax(depth_data)
        elif nan_mode == NAN_MODE_MEAN: swap_value = np.nanmean(depth_data)
        elif nan_mode == NAN_MODE_MEDIAN: swap_value = np.nanmedian(depth_data)
        else: swap_value = 0

        depth_data = np.nan_to_num(depth_data, swap_value)




def convert_file_bin2png(path_file: str, nan_mode: int = NAN_MODE_MIN) -> np.ndarray:
    f = open(path_file, "rb")
    compressed_array = f.read()
    f.close()
    pcd = blosc.unpack_array(compressed_array)
    return convert_pcd2png(pcd, nan_mode)

