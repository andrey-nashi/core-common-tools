import cv2
import numpy as np

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