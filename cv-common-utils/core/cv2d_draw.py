import cv2
import numpy as np

class ImageVisualizer:

    @staticmethod
    def draw_bbox(image: np.ndarray, bbox: list, color: list = (0, 255, 0), text: str = None, brush_size: int = 2) -> np.ndarray:
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
        cv2.rectangle(drawn_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color[0], color[1], color[2]), brush_size)
        if text is not None:
            cv2.putText(drawn_img, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (color[0], color[1], color[2]), 2, cv2.LINE_AA)
        return drawn_img

    @staticmethod
    def draw_boxes(image: np.ndarray, boxes: list, color: list = (0, 255, 0), text: str = None, brush_size: int = 2):
        drawn_img = image.copy()
        for i in range(0, len(boxes)):
            box = boxes[i]
            label = None
            if text is not None:
                if isinstance(text, str):
                    label = text
                elif isinstance(text, list) and len(boxes) == len(text):
                    label = text[i]

            drawn_img = ImageVisualizer.draw_bbox(drawn_img, box, color, str(label), brush_size)

        return drawn_img

    @staticmethod
    def draw_mask(image: np.ndarray, mask: np.ndarray, color: list, transparency: float = 0.2):
        mask_h, mask_w = mask.shape
        image_h = image.shape[0]
        image_w = image.shape[1]

        if mask_h != image_h or mask_w != image_w:
            mask = cv2.resize(mask, (image_w, image_h))

        mx = np.zeros((image_h, image_w, 3), dtype=np.uint8)

        mx[mask > 0] = color

        image = cv2.addWeighted(image, 1, mx, transparency, 0)

        return image

    @staticmethod
    def draw_mask_contour(image: np.ndarray, mask: np.ndarray, color: list, brush_size: int = 1):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.drawContours(image, contours, -1, color, brush_size)

        return image

    @staticmethod
    def draw_masks_horizontal(image: np.ndarray, mask_list: list, color_list: list, resize=None):
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


            image_out = ImageVisualizer.draw_mask(np.copy(image), mask, color_list[i])
            image_out = ImageVisualizer.draw_mask_contour(image_out, mask, color_list[i])

            concatenation_list.append(image_out)

        return cv2.hconcat(concatenation_list)
