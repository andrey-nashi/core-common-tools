import cv2
import numpy as np


def cv2d_mask_fill_holes(mask: np.ndarray, f_iteration_count: int = 5, f_check_change: bool = False) -> np.ndarray:
    """
    Fill holes in connected objects in a given mask
    :param mask: binary mask [h,w] where (i,j) in [0,255]
    :param f_iteration_count: number of iterations, the higher, the slower is the algorithm
    :param f_check_change: check whether change has occurred between iterations, allows early stop
    :return: processed binary mask
    """
    mask_org = mask.copy()
    change_flag = False
    for i in range(1, 1 + 2 * f_iteration_count, 2):
        if i > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (i, i))
            mask = cv2.morphologyEx(mask_org, cv2.MORPH_CLOSE, kernel)
        mask_cls = mask.copy()
        contour, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contour, -1, 255, -1)
        mask_diff = mask - mask_cls
        change_flag = np.any(mask_diff == 255)

        if change_flag:
            if f_check_change:
                _, labels = cv2.connectedComponents(mask)
                val_unique = np.unique(labels[mask_diff != 0])
                if len(val_unique) != labels.max():
                    change_flag = False
            break
    else:
        mask = mask_org
    return mask


def cv2d_mask_close_contours(mask: np.ndarray, distance_min: int = 10, distance_max: int = 50, step: int = 2) -> np.ndarray:
        """
        If the mask has an object with high convexity, then this method allows to
        enclose such object. Algorithm is based on distance transform and could be slow if
        min, max and step arguments are chosen poorly. For each individual connected component,
        distance transformation is applied iteratively. Then it is tested - how many objects
        are formed by isolines with the specific distance being tested. If the original object
        is an non-closed contour - then for low values of distance, there will be one such isoline
        object, for higher values - two. The distance at which there is a change - is the threshold
        distance. When this distance is found, then what is left - grab the isoline object that represents
        the inner part of the original contour, expand it and fill everything.
        :param mask: numpy array - mask with annotations. Enclosed contours should be filled.
        :param distance_min: minimum distance to use for tests
        :param distance_max: maximum distance to use for tests
        :param step: step for iterate in a loop from distance_min to distance_max
        :return: restored mask
        """
        output = mask.copy()

        r, label_map_global = cv2.connectedComponents(mask)

        for i in range(1, np.max(label_map_global) + 1):
            one_object_mask = np.zeros(label_map_global.shape, dtype=np.uint8)
            one_object_mask[label_map_global == i] = 255

            MAGIC_NUMBER = 2
            image = 255 - np.copy(one_object_mask)
            image = cv2.distanceTransform(image, cv2.DIST_L2, 5)

            is_found_hole = False
            threshold_distance = 0
            contour_map = None

            # ---- Attempt to find the threshold distance, itterating over
            # ---- various distances in the specified range.
            for d in range(distance_min, distance_max, step):
                threshold_distance = d
                x = image.copy()
                x[x < threshold_distance] = 0
                x[x > threshold_distance + MAGIC_NUMBER] = 0
                x[x != 0] = 255
                x = x.astype(np.uint8)

                r, contour_map = cv2.connectedComponents(x)
                object_count = np.max(contour_map)
                if object_count > 1:
                    is_found_hole = True
                    break
            if not is_found_hole:
                output[one_object_mask != 0] = 255

            # ---- Determine which contour is inner which is outer
            inner_index = 1
            if len(np.argwhere(contour_map == 1)) > len(np.argwhere(contour_map == 2)):
                inner_index = 2

            # ---- Fill insides, and all between
            x = np.zeros(label_map_global.shape, dtype=np.uint8)
            contour = np.argwhere(contour_map == inner_index)
            contour = np.array([[c[1], c[0]] for c in contour])
            if len(contour) != 0:
                x = cv2.drawContours(x, [contour], -1, 255, -1)
                x = cv2.distanceTransform(255 - x, cv2.DIST_L2, 5)
                x[x > threshold_distance + MAGIC_NUMBER] = 255
                x[x <= threshold_distance + MAGIC_NUMBER] = 0
                x = 255 - x

                output = output + x

        output[output > 255] = 255
        output[output < 0] = 0
        output = output.astype(np.uint8)

        return output


def cv2d_denoise_mask(mask: np.ndarray, threshold: float = 0.5, kernel_size: int = 5, minimum_area_size: int = 0) -> np.ndarray:
    """
    For the given image applies gaussian filter, removes positive regions that are less than specified
    area and fills holes in all the remaining regions.
    :param mask: input numpy array, a mask of elements [0,1] or [0, 255]
    :param threshold: binarization threshold
    :param kernel_size: kernel of the gaussian
    :param minimum_area_size: minimum number of pixels for an area to be kept
    :return:
    """

    # ---- Blur the given maks with Gaussian filter
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    # ---- Binarize the mask using the given threshold
    mask = (mask >= threshold).astype(mask.dtype)


    # ---- Fill holes
    score2 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(score2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        cv2.drawContours(mask, [i], 0, 1, -1)

    # ---- Remove regions with areas less than specified
    score2 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(score2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        if cv2.contourArea(i) < minimum_area_size:
            cv2.drawContours(mask, [i], -1, 0, -1)

    # ---- Closing = dilate + erode. kernel size is tentative (its effect is limited though)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

