import os.path

import cv2
import numpy as np
import albumentations as alb
from core.datasets.ds_segmentation import DatasetSegmentationBinary
from core.models.seg_smp.model_pl import SmpModel_Light


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



path_model = "./lightning_logs/version_5/checkpoints/epoch=29-step=1170.ckpt"
model = SmpModel_Light.load_from_checkpoint(path_model)
transform_val = alb.Compose([alb.Resize(512, 512, p=1, always_apply=True)])
"""
transform_val = alb.Compose([alb.Resize(512, 512, p=1, always_apply=True)])
path_test = "/home/andrey/Dev/tote-data/td3-test.json"
dataset = DatasetSegmentationBinary(transform_val)
dataset.serialize_from_json(path_test, "/home/andrey/Dev/tote-data")
dataset.switch_to_tensor(False)

path_test_out = "/home/andrey/Dev/tote-data/test-out-3"


for i in range(0, len(dataset)):
    img, mask_gt = dataset.__getitem__(i)

    mask_pr = model.predict(img)
    mask_pr = mask_pr * 255

    output = draw_mask(img, mask_pr, (0, 255, 0), 0.5)
    cv2.imwrite(os.path.join(path_test_out, str(i) + ".png"), output)
"""

img = cv2.imread("/home/andrey/Dev/tote-data/vc.jpg")
x = transform_val(image=img)
img = x["image"]
mask_pr = model.predict(img)
mask_pr = mask_pr * 255
output = draw_mask(img, mask_pr, (0, 255, 0), 0.5)
cv2.imwrite("/home/andrey/Dev/tote-data/vc-out.jpg", output)

