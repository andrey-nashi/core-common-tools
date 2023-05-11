import os
import cv2
import json

from .ds_base import convert_image2tensor
from .ds_base import AbstractDataset


class DatasetSegmentationBinary(AbstractDataset):

    SERIAL_KEY_DATASET = "dataset"
    SERIAL_KEY_IMAGE = "image"
    SERIAL_KEY_MASK = "mask"

    def __init__(self, transform_func: callable = None):
        """
        Dataset for binary segmentation task where mask is given as 0/255 image
        :param transform_func: should be a callable method with arguments 'image' and 'mask'
        which would return a dictionary with 'image' and 'mask' fields.
        Albumentations style transformation function
        """
        super().__init__(transform_func)
        self.path_root_dir = None

    def serialize_from_json(self, path_file: str, path_root_dir: str = None):
        f = open(path_file, "r")
        data = json.load(f)
        f.close()

        self.path_root_dir = path_root_dir

        for sample in data[self.SERIAL_KEY_DATASET]:
            path_image = sample[self.SERIAL_KEY_IMAGE]
            path_mask = sample[self.SERIAL_KEY_MASK]

            self.samples_table.append({self.SERIAL_KEY_IMAGE: path_image, self.SERIAL_KEY_MASK: path_mask})

    def serialize_to_json(self, path_file: str, **kwargs):
        f = open(path_file, "w")
        json.dump({self.SERIAL_KEY_DATASET: self.samples_table}, f)
        f.close()

    def __getitem__(self, sample_index: int):
        path_image = self.samples_table[sample_index][self.SERIAL_KEY_IMAGE]
        path_mask = self.samples_table[sample_index][self.SERIAL_KEY_MASK]

        # ---- Dataset with both image and mask are specified
        if path_image is not None and path_mask is not None:

            if self.path_root_dir is not None:
                path_image = os.path.join(self.path_root_dir, path_image)
                path_mask = os.path.join(self.path_root_dir, path_mask)

            image = cv2.imread(path_image)
            mask = cv2.imread(path_mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask / 255

            transformed = self.transform_func(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

            if self.is_to_tensor:
                transformed_image = convert_image2tensor(transformed_image)
                transformed_mask = convert_image2tensor(transformed_mask)

            return transformed_image, transformed_mask

        # ---- Dataset with only image specified, can be used in testing
        if path_image is not None and path_mask is None:
            if self.path_root_dir is not None:
                path_image = os.path.join(self.path_root_dir, path_image)

            image = cv2.imread(path_image)
            transformed = self.transform_func(image=image)
            transformed_image = transformed['image']

            if self.is_to_tensor:
                transformed_image = convert_image2tensor(transformed_image)

            return transformed_image, None

