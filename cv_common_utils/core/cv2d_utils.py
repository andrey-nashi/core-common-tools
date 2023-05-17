import cv2
import numpy as np

def cv2d_swap_color(image: np.ndarray, color_old: list, color_new: list) -> np.ndarray:
    assert len(color_old) == 3
    assert len(color_new) == 3
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    img_b, img_g, img_r = cv2.split(image)
    mask = np.logical_and(img_r == color_old[0], np.logical_and(img_g == color_old[1], img_b == color_old[2]))

    img_r[mask] = color_new[0]
    img_g[mask] = color_new[1]
    img_b[mask] = color_new[2]

    output = cv2.merge([img_b, img_g, img_r])
    return output
