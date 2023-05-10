import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class DatasetSegmentation(Dataset):
    def __init__(self):
        self.transform = None
        self.table = []

    def load_from_json(self, path_file):
        f = open(path_file, "r")
        data = json.load(f)
        f.close()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label