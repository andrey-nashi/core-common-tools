import numpy as np
import torch

from torch.utils.data import Dataset

def convert_image2tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array that represents an image like (H,W) or (H,W,3) into tensor like (1,H,W) or (3,H,W).
    :param image: image to be transformed
    :return: Tensor or None if dimensions don't match
    """
    if image.ndim == 3:
        transformed_image = torch.from_numpy(image)
        transformed_image = transformed_image.permute(2, 0, 1)
        return transformed_image
    if image.ndim == 2:
        transformed_image = torch.unsqueeze(image, 0)
        return transformed_image
    return None



class AbstractDataset(Dataset):

    def __init__(self, transform_func: callable = None):
        self.transform_func = transform_func
        self.is_to_tensor = True
        self.samples_table = []

    def switch_to_tensor(self):
        self.is_to_tensor = not self.is_to_tensor

    def __len__(self):
        return len(self.table)

    def __getitem__(self, sample_index: int):
        if sample_index < len(self.samples_table):
            return self.samples_table
        else:
            return None

    def __add__(self, dataset):
        assert type(self) == type(dataset)

        output = type(self)(self.transform_func)
        for attribute_name in vars(self):
            if not

        return a




x = AbstractDataset()
x.samples_table = [0, 1, 2]

y = AbstractDataset()
y.samples_table = [3, 6]

z = x + y
print(z.samples_table)

print(x.samples_table, y.samples_table)

