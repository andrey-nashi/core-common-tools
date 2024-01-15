import cv2
import numpy as np
import math

def cv2d_swap_color(image: np.ndarray, color_old: list, color_new: list) -> np.ndarray:
    """
    Swap all pixels of the given color to new color
    :param image: numpy array representing an image
    :param color_old: target color given as list [R,G,B]
    :param color_new: new color given as list [R,G,B]
    :return: new generated image
    """
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

def cv2d_add_noise_gaussian(image: np.ndarray, mean: int = 0, sigma: int = 25) -> np.ndarray:
    """
    Add gaussian noise to the given image
    :param image: numpy array representing an image
    :param mean: gaussian noise mean value
    :param sigma: gaussian noise sigma value
    :return: new generated image
    """
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)

def cv2d_add_noise_sp(image: np.ndarray, salt_prob: float = 0.02, pepper_prob: float = 0.02):
    """
    Add salt and pepper noise to the given image
    :param image: numpy array representing an image
    :param salt_prob: probability of 'salt' noise
    :param pepper_prob: probability of 'pepper' noise
    :return: new generated image
    """
    row = image.shape[0]
    col =image.shape[1]

    noisy = np.copy(image)

    # ---- Salt noise
    salt = np.random.rand(row, col) < salt_prob
    noisy[salt, :] = 255

    # ---- Pepper noise
    pepper = np.random.rand(row, col) < pepper_prob
    noisy[pepper, :] = 0

    return noisy.astype(np.uint8)


def cv2d_remove_blobs(mask: np.ndarray, blob_min_size: int, mode: int = 0):
    """
    Remove all blobs that have less pixels than the given threshold
    :param mask: binary mask 0|255
    :param blob_min_size: min size of the blob
    :param mode - size check mode
    - MODE_AREA=0 (S < T)
    - MODE_RESOLUTION=1 (w & h < T)
    - MODE_DIAGONAL=2 (sq(w2 + h2) < T)
    :return: binary mask with small blobs removed
    """
    MODE_AREA = 0
    MODE_RESOLUTION = 1
    MODE_DIAGONAL = 2

    mask_output = mask.copy()
    mask_output = mask_output.astype(np.uint8)

    ret, labels = cv2.connectedComponents(mask_output)
    for label in range(1, ret):
        xy = np.argwhere(labels == label).T

        blob_area = len(xy[0])
        bbox_x_min = int(min(xy[1]))
        bbox_y_min = int(min(xy[0]))
        bbox_x_max = int(max(xy[1]))
        bbox_y_max = int(max(xy[0]))

        blob_h = bbox_x_max - bbox_x_min
        blob_w = bbox_y_max - bbox_y_min
        blob_d = math.sqrt(blob_h * blob_h + blob_w * blob_w)

        if mode == MODE_AREA and blob_area < blob_min_size:
            mask_output[labels == label] = 0
        if mode == MODE_RESOLUTION and blob_w < blob_min_size and blob_h < blob_min_size:
            mask_output[labels == label] = 0
        if mode == MODE_DIAGONAL and blob_d < blob_min_size:
            mask_output[labels == label] = 0

    return mask_output