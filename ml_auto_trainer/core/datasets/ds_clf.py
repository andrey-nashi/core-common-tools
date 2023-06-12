import os
import cv2
import json
import numpy as np
from .ds_base import convert_image2tensor, normalize_numpy, convert_list2tensor
from .ds_base import AbstractDataset


class DatasetClassification(AbstractDataset):

    SERIAL_KEY_DATASET = "dataset"
    SERIAL_KEY_IMAGE = "image"
    SERIAL_KEY_LABELS = "labels"

    BK_IMAGE = "image"
    BK_LABELS = "labels"
    BATCH_KEYS = [BK_IMAGE, BK_LABELS]

    NORMALIZE_NONE = 0
    NORMALIZE_255 = 1
    NORMALIZE_MINMAX = 2

    def __init__(self, transform_func: callable = None, norm_image_mode: bool = NORMALIZE_255):
        """
        Dataset for classification tasks
        :param transform_func: should be a callable method with arguments 'image' and 'mask'
        which would return a dictionary with 'image' and 'mask' fields.
        Albumentations style transformation function
        :param norm_image_mode: normalization mode
        """
        super().__init__(transform_func)
        self.path_root_dir = None
        self.norm_image_mode = norm_image_mode

    def load_from_json(self, path_file: str, path_root_dir: str = None) -> bool:
        """
        Load classification dataset from a JSON file specified by path
        :param path_file: absolute path to JSON file, must have the following format.
        {"dataset": [{"image": <path_to_image>, "label": [$l, $l]}, {...}]}
        <label> - is an N-dim vector
        :param path_root_dir: path to root directory with images or masks, it will
        be used as prefix to all paths in the JSON dataset
        :return: True if success, False if failed
        """
        try:
            f = open(path_file, "r")
            data = json.load(f)
            f.close()

            self.path_root_dir = path_root_dir

            for sample in data[self.SERIAL_KEY_DATASET]:
                path_image = sample[self.SERIAL_KEY_IMAGE]
                labels = sample[self.SERIAL_KEY_LABELS]

                self.samples_table.append({self.SERIAL_KEY_IMAGE: path_image, self.SERIAL_KEY_LABELS: labels})
            return True
        except Exception:
            return False

    def save_to_json(self, path_file: str, **kwargs):
        """
        Save this dataset into a JSON file by the given path.
        :param path_file: path to the output JSON file
        :param kwargs:
        :return: True if success, False if failed
        """
        try:
            f = open(path_file, "w")
            json.dump({self.SERIAL_KEY_DATASET: self.samples_table}, f)
            f.close()
            return True
        except Exception:
            return False

    def __getitem__(self, sample_index: int):
        """
        Overloaded method to be used by torch.Dataloader. Will output a list of two elements
        :param sample_index: index of the sample
        :return: list of two items
        - [image, labels] where both image and labels are numpy arrays or Tensors (depending on the is_to_tensor flag)
        - [image, None]
        """
        path_image = self.samples_table[sample_index][self.SERIAL_KEY_IMAGE]
        labels = self.samples_table[sample_index][self.SERIAL_KEY_LABELS]

        # ---- Dataset with both image and labels are specified
        if path_image is not None and labels is not None:

            if self.path_root_dir is not None:
                path_image = os.path.join(self.path_root_dir, path_image)

            image = cv2.imread(path_image)

            transformed = self.transform_func(image=image)
            transformed_image = transformed[self.BK_IMAGE]
            transformed_image = normalize_numpy(transformed_image, self.norm_image_mode)
            transformed_labels = np.array(labels)

            if self.is_to_tensor:
                transformed_image = convert_image2tensor(transformed_image)
                transformed_labels = convert_list2tensor(transformed_labels)

            return transformed_image, transformed_labels

        # ---- Dataset with only image specified, can be used in testing
        if path_image is not None and labels is None:
            if self.path_root_dir is not None:
                path_image = os.path.join(self.path_root_dir, path_image)

            image = cv2.imread(path_image)
            transformed = self.transform_func(image=image)
            transformed_image = transformed[self.BK_IMAGE]

            transformed_image = normalize_numpy(transformed_image, self.norm_image_mode)

            if self.is_to_tensor:
                transformed_image = convert_image2tensor(transformed_image)

            return transformed_image, None

