import os.path

import cv2
import numpy as np
import albumentations as alb
from core.datasets.ds_segmentation import DatasetSegmentationBinary
from core.models.seg_smp.model_pl import SmpModel_Light

#-----------------------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------------------
def run_on_dataset(path_model, path_test, path_dir_root, path_test_out, image_size):
    model = SmpModel_Light.load_from_checkpoint(path_model)

    transform_test = alb.Compose([alb.Resize(image_size, image_size, p=1, always_apply=True)])
    dataset = DatasetSegmentationBinary(transform_test)
    dataset.load_from_json(path_test, path_dir_root)
    dataset.switch_to_tensor(False)

    for i in range(0, len(dataset)):
        img, mask_gt = dataset.__getitem__(i)
        mask_pr = model.predict(img)
        output = draw_mask(img, mask_pr, (0, 255, 0), 0.5)
        cv2.imwrite(os.path.join(path_test_out, str(i) + ".png"), output)


def run_on_image(path_model, path_image, path_test_out, image_size):
    model = SmpModel_Light.load_from_checkpoint(path_model)
    transform_test = alb.Compose([alb.Resize(image_size, image_size, p=1, always_apply=True)])

    image = cv2.imread(path_image)
    result = transform_test(image=image)
    image = result["image"]

    mask_pr = model.predict(image)
    output = draw_mask(image, mask_pr, (0, 255, 0), 0.5)
    cv2.imwrite(path_test_out, output)

#-----------------------------------------------------------------------------------------

if __name__ == '__main__':
    path_model = "./lightning_logs/version_7/checkpoints/epoch=0-step=39.ckpt"
    path_test = "/home/andrey/Dev/tote-data/td2-test.json"
    path_dir_root = "/home/andrey/Dev/tote-data/"
    path_test_out = "/home/andrey/Dev/tote-data/test-out-4"
    image_size = 512
    run_on_dataset(path_model, path_test, path_dir_root, path_test_out, 512)

    path_image = "/home/andrey/Dev/tote-data/example.jpg"
    path_test_out = "/home/andrey/Dev/tote-data/example-out.jpg"
    run_on_image(path_model, path_image, path_test_out, image_size)
