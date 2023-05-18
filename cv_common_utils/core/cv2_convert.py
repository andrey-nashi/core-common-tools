import random
import cv2
import numpy as np


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


def cv2d_convert_polygon2mask(width: int, height: int, vertex_list: list, intensity: int = 255) -> np.ndarray:
    """
    Generate a mask of resolution (width, height) and draw a polygon specified by the list of vertices.
    :param width: width of the output mask
    :param height: height of the output mask
    :param vertex_list: list of polygon vertices [x,y]
    :param intensity: value of the pixels of the generated polygon
    :return: numpy array representing the mask
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    v = np.array(vertex_list, np.int32)
    v = v.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [v], (intensity))
    return mask


def cv2d_convert_polygons2mask(width: int, height: int, polygon_list: list, intensity: int = 255) -> np.ndarray:
    """
    Generate a mask of resolution (width, height) and draw a polygons specified in the given list
    :param width: width of the output mask
    :param height: height of the output mask
    :param polygon_list: list of polygons, each polygon is a list of verices [x,y]
    :return: numpy array representing the mask
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygon_list:
        v = np.array(polygon, np.int32)
        v = v.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [v], (intensity))
    return mask


def cv2d_convert_mask2polygon(mask: np.array) -> list:
    """
    Convert a binary mask to polygon (image should contain only 1 object)
    :param mask: binary mask as a numpy array [H,W], each element is 0|1
    :return: list of polygons as an array of vertices [ <[[x,y], [x,y]]>, <[[x,y], [x,y]]> ]
    """
    output = []

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return output

    for contour_id in range(0, len(contours)):
        hull = cv2.convexHull(contours[0], False)
        hull = hull.squeeze(1)
        l, d = hull.shape

        polygon = []
        for i in range(0, l):
            x = hull[i][0]
            y = hull[i][1]
            polygon.add([x,y])

        output.append(polygon)

    return output


def cv2d_convert_pix2label(image: np.ndarray, color_codes: dict, one_hot=False, dims=2) -> np.ndarray:
    """
    Convert a 3-channel image to a label map using a color LUT.
    :param image: a 3-channel numpy array that represents an image
    :param color_codes: a dictionary that defines conversion between class labels and colors.
    For binary segmentation task the table could defined as: {0: [0, 0, 0], 1: [255, 255, 255]}
    :param one_hot: if set to True the produced label map will have one hot encoding
    :return: a numpy array, label map
    """

    if one_hot:
        result = np.zeros((*image.shape[:dims], len(color_codes)), dtype=int)
        for idx, rgb in color_codes.items():
            x = np.zeros(len(color_codes))
            x[idx] = 1
            result[np.asarray(image == rgb).all(dims)] = x
    else:
        ch = None
        if isinstance(random.choice(list(color_codes.keys())), (list, tuple)):
            assert len(set([len(k) for k in color_codes])) == 1, color_codes
            ch = len(random.choice(list(color_codes.keys())))
        result = np.zeros(image.shape[:dims], dtype=int)
        result = result if ch is None else np.stack([result]*ch, -1)
        for idx, rgb in color_codes.items():
            result[np.asarray(image == rgb).all(dims)] = idx
        result = np.expand_dims(result, axis=-1) if ch is None else result
    return result

def cv2d_convert_label2pix(label_map, color_codes, one_hot=False):
    """
    Convert a label map into an image using the given color LUT.
    :param label_map: a 1-channel or N-channel numpy array
    :param color_codes: a dictionary that defines conversion between class labels and colors.
    For binary segmentation task the table could defined as: {0: [0, 0, 0], 1: [255, 255, 255]}
    :param one_hot: if set to True the produced label map will have one hot encoding
    :return: an numpy array, image
    """
    result = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype='uint8')
    for idx, rgb in color_codes.items():
        if one_hot:
            x = np.zeros(len(color_codes))
            x[idx] = 1
            result[np.asarray(label_map == x).all(2)] = rgb
        else:
            result[(label_map == idx)] = rgb
    return result
