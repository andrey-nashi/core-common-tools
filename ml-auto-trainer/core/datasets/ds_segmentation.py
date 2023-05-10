import os
import cv2
import json
from torch.utils.data import Dataset

class DatasetSegmentationBinary(Dataset):
    def __init__(self, transform):
        self.path_root_dir = None
        self.transform = transform
        self.table = []
        self.is_to_tensor = True

    def serialize_from_json(self, path_file, path_root_dir: str):
        f = open(path_file, "r")
        data = json.load(f)
        f.close()

        self.path_root_dir = path_root_dir

        for sample in data["dataset"]:
            path_image = sample["image"]
            path_mask = sample["mask"]

            self.table.append({"image": path_image, "mask": path_mask})

    def switch_flag_tt(self):
        self.is_to_tensor = not self.is_to_tensor

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        path_absolute_img = os.path.join(self.path_root_dir, self.table[idx]["image"])
        path_absolute_mask = os.path.join(self.path_root_dir, self.table[idx]["mask"])

        image = cv2.imread(path_absolute_img)

        mask = cv2.imread(path_absolute_mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        mask = mask / 255

        transformed = self.transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        if self.is_to_tensor:
            transformed_image = torch.from_numpy(transformed_image)
            transformed_mask = torch.from_numpy(transformed_mask)
            transformed_mask = torch.unsqueeze(transformed_mask, 0)

            #print(transformed_image.size(), transformed_mask.size(), "++++++++++++")
            transformed_image = transformed_image.permute(2, 0, 1)

            #print(transformed_image.size(), transformed_mask.size(), "<<<<<<<<<<<<<<<<<<<<")

        return transformed_image, transformed_mask

