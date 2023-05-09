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

def cv2d_convert_bbox2mask(height: int, width: int, bbox_list: list, label_list: list = None, label_target: int = None, score_list: list = None) -> np.ndarray:
    """
    Generate mask of size [h,w] with rectangular roi objects defined by bounding box coordinates
    :param height: height of the output mask
    :param width: width of the output mask
    :param bbox_list: list of boxes [[xmin, ymin, xmax, ymax]]
    :param label_list: list of labels [id, id] or None
    :param label_target: if not None, then pick only boxes with specific labels
    :param score_list: if not None, then generate intensity encoded mask
    :return: numpy array representing a mask
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for index in range(0, len(bbox_list)):
        box = bbox_list[index]
        label = label_list[index]

        if score_list is None: score = 1
        else: score = score_list[index]

        if (label_list is None) or (label_target is None) or (label_target == label):
            minx = min(box[1], box[3])
            maxx = max(box[1], box[3])
            miny = min(box[0], box[2])
            maxy = max(box[0], box[2])

            mask[minx:maxx, miny:maxy] = int(255 * score)

    return mask

def cv2d_convert_mask2bbox(mask: np.ndarray, threshold_probability: float = None, threshold_size: int = None) -> list:
    """
    Detecte connected components on a given grayscale mask, calculate bounding boxes and scores. Score is
    the average intensity of the object / 255. Before executing connected components the mask will be
    transformed into binary one using the threshold_probability, and objects less size than
    threshold_size will not be used.
    :param mask: a numpy array, if BGR is given will be automatically converted to grayscale
    :param threshold_probability: threshold to apply binarization of the given mask, if not given '128' will be used
    :param threshold_size: objects less than this threshold will not be taken into account
    :return: list of dictionaries {"bbox": [x_min, y_min, x_max, y_max], "score": s} or None
    """

    # ---- Check arguments
    if threshold_probability <= 0: return None
    if threshold_probability > 1: return None
    if threshold_size < 0: return None
    if threshold_size is None: threshold_size = 0

    # ---- Convert to grayscale mask if needed
    if len(mask.shape) == 3:
        mask_x = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_x = mask.copy()

    # ---- Make binary mask applying given probability
    if threshold_probability is not None:
        r, mask_binary = cv2.threshold(mask_x, int(threshold_probability * 255), 255, cv2.THRESH_BINARY)
    else:
        r, mask_binary = cv2.threshold(mask_x, 128, 255, cv2.THRESH_BINARY)

    # ---- Apply connected components
    r, object_map = cv2.connectedComponents(mask_binary)

    output = []

    for object_id in range(1, np.max(object_map) + 1):
        object_xy = np.argwhere(object_map == object_id).T
        object_size = len(object_xy[0])
        object_score = np.mean(mask_x[object_map == object_id]) / 255

        if object_size > threshold_size:
            bbox_x_min = int(min(object_xy[1]))
            bbox_y_min = int(min(object_xy[0]))
            bbox_x_max = int(max(object_xy[1]))
            bbox_y_max = int(max(object_xy[0]))
            score = float(object_score)

            output.append({"bbox": [bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max], "score": score})

    return output