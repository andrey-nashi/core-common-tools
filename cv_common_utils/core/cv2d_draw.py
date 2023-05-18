import cv2
import numpy as np


def cv2d_draw_bbox(image: np.ndarray, bbox: list, color: list = (0, 255, 0), text: str = None, brush_size: int = 2) -> np.ndarray:
    """
    Draw bounding box on a given image and annotate it with some text
    :param image: an input image, 3 channels
    :param bbox: list of bounding box coordinates [xmin, ymin, xmax, ymax]
    :param color: color to draw the box with (R,G,B)
    :param text: a text string to annotate bounding box with
    :param brush_size: thickness of the brush to draw the box with
    :return: a numpy array - an image with the drawn bounding box
    """
    drawn_img = image.copy()
    cv2.rectangle(drawn_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color[0], color[1], color[2]),
                  brush_size)
    if text is not None:
        cv2.putText(drawn_img, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (color[0], color[1], color[2]), 2, cv2.LINE_AA)
    return drawn_img


def cv2d_draw_boxes(image: np.ndarray, boxes: list, color: list = (0, 255, 0), text: str = None, brush_size: int = 2) -> np.ndarray:
    """
    Draw bounding boxes with the specified color and annotated with text labels.
    :param image: numpy image of [H,W,C]
    :param boxes: list of boxes as [[xmin, ymin, xmax, ymax]]
    :param color: color as [r,g,b]
    :param text: optional annotations, a string or list of strings
    :param brush_size: thickness of the bounding box borders
    :return: a numpy array - an image with the drawn bounding boxes
    """
    drawn_img = image.copy()
    for i in range(0, len(boxes)):
        box = boxes[i]
        label = None
        if text is not None:
            if isinstance(text, str):
                label = text
            elif isinstance(text, list) and len(boxes) == len(text):
                label = text[i]

        drawn_img = cv2d_draw_bbox(drawn_img, box, color, str(label), brush_size)

    return drawn_img


def cv2d_draw_mask(image: np.ndarray, mask: np.ndarray, color: list, transparency: float = 0.2) -> np.ndarray:
    """
    Draw mask with specified color on a given image
    :param image: numpy image of [H,W,C]
    :param mask: mask of [H,W] with [0,X] where X any value will be treated as mask
    :param color: color to draw mask with [R,G,B]
    :param transparency: transparency of the mask to draw
    :return: numpy array of the new image
    """
    mask_h, mask_w = mask.shape
    image_h = image.shape[0]
    image_w = image.shape[1]

    if mask_h != image_h or mask_w != image_w:
        mask = cv2.resize(mask, (image_w, image_h))

    mx = np.zeros((image_h, image_w, 3), dtype=np.uint8)

    mx[mask > 0] = color

    image = cv2.addWeighted(image, 1, mx, transparency, 0)

    return image


def cv2d_draw_mask_contour(image: np.ndarray, mask: np.ndarray, color: list, brush_size: int = 1) -> np.ndarray:
    """
    Draw mask contours on a given image with specified color
    :param image: numpy image of [H,W,C]
    :param mask: mask of [H,W] with [0,X] where X any value will be treated as mask
    :param color: color to draw mask with [R,G,B]
    :param brush_size: thickness of the edges
    :return: new mask
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, color, brush_size)
    return image


def cv2d_draw_masks_horizontal(image: np.ndarray, mask_list: list, color_list: list, resize=None) -> np.ndarray:
    """
    Generate a new image like [image, (image & mask), (image & mask), ... ] - image and image with overlayed masks.
    :param image: a numpy image
    :param mask_list: list of masks [<np.ndarray>, <np.ndarray>, ... ]
    :param color_list: list of colors [[R,G,B]] for each mask
    :param resize: resolution of (W,H), if not None will forcefully resize input image and masks
    :return: numpy array of [H, N*W]
    """
    concatenation_list = []

    if resize is not None:
        image = cv2.resize(image, resize)

    h = image.shape[0]
    w = image.shape[1]

    concatenation_list.append(image)
    for i in range(0, len(mask_list)):
        mask = mask_list[i]
        mask = cv2.resize(mask, (w, h))
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        r, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

        image_out = cv2d_draw_mask(np.copy(image), mask, color_list[i])
        image_out = cv2d_draw_mask_contour(image_out, mask, color_list[i])

        concatenation_list.append(image_out)

    return cv2.hconcat(concatenation_list)


def cv2d_draw_error_map(mask_gt: np.array, mask_pr: np.array, color_table: dict = {"tp": [0, 255, 0], "fp": [0, 0, 255], "fn": [255, 0, 0]}):
    """
    Generate an error map for the specific ground truth and predicted masks.
    Resolution of the masks should be the same.
    :param mask_gt: an opencv image, binary mask (0, 255)
    :param mask_pr: an opencv image, binary mask (0, 255)
    :param color_table: table of colors corresponding to TP, FP, FN given as dictionary
    {"tp": [R,G,B], "fp": [R,G,B], "fn": [R,G,B]}
    :return: an opencv image, the error map
    """
    if mask_gt.shape != mask_pr.shape:
        return None

    resolution = mask_gt.shape
    error_map = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)

    # ---- Below encode the TP,FP,FN with BGR colors
    # ---- TP
    xy = np.argwhere(np.bitwise_and(mask_pr == 255, mask_gt == 255)).T
    error_map[xy[0], xy[1], 0] = color_table["tp"][2]
    error_map[xy[0], xy[1], 1] = color_table["tp"][1]
    error_map[xy[0], xy[1], 2] = color_table["tp"][0]
    # ---- FP
    xy = np.argwhere(np.bitwise_and(mask_pr == 255, mask_gt == 0)).T
    error_map[xy[0], xy[1], 0] = color_table["fp"][2]
    error_map[xy[0], xy[1], 1] = color_table["fp"][1]
    error_map[xy[0], xy[1], 2] = color_table["fp"][0]
    # ---- FN
    xy = np.argwhere(np.bitwise_and(mask_pr == 0, mask_gt == 255)).T
    error_map[xy[0], xy[1], 0] = color_table["fn"][2]
    error_map[xy[0], xy[1], 1] = color_table["fn"][1]
    error_map[xy[0], xy[1], 2] = color_table["fn"][0]

    return error_map
