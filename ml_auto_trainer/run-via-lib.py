import cv2
import torch
import albumentations as alb
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from core.datasets import DatasetSegmentationBinary
from core.models.seg_unet2 import UpashUnet_Light
from core.models.seg_smp import SmpModel_Light
from core.models import ModelFactory

def run_train(
        path_json_train: str,
        path_json_val: str,
        path_dir_root: str,
        image_size: int,
        nn_architecture: str,
        nn_encoder: str,
        nn_in_channels: int,
        nn_out_channels: int
        ):

    # ---- Define transformations for train/val
    transform_train = alb.Compose([
        alb.RandomRotate90(),
        alb.Flip(),
        alb.Transpose(),
        alb.IAAPerspective(),
        alb.OneOf([alb.IAAAdditiveGaussianNoise(), alb.GaussNoise()], p=0.2),
        alb.OneOf([alb.MotionBlur(p=0.2), alb.MedianBlur(blur_limit=3, p=0.1), alb.Blur(blur_limit=3, p=0.1)], p=0.2),
        alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        alb.OneOf([alb.OpticalDistortion(p=0.3), alb.GridDistortion(p=0.1), alb.IAAPiecewiseAffine(p=0.3)], p=0.2),
        alb.Resize(image_size, image_size, p=1, always_apply=True),
    ])

    transform_val = alb.Compose([alb.Resize(image_size, image_size, p=1, always_apply=True)])

    # ---- Datasets and loaders
    dataset_train = DatasetSegmentationBinary(transform_train)
    dataset_train.load_from_json(path_json_train, path_dir_root)
    dataset_valid = DatasetSegmentationBinary(transform_val)
    dataset_valid.load_from_json(path_json_val, path_dir_root)
    train_dataloader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(dataset_valid, batch_size=4, shuffle=False, num_workers=4)

    loss_func = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)
    #model = UpashUnet_Light(nn_in_channels, nn_out_channels, loss_func, 0.01)
    model = SmpModel_Light("Unet", "resnet50", 3, 1, loss_func, activation="sigmoid")
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=30)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


def run_test(path_json_val, path_dir_root, path_model):
    transform_val = alb.Compose([alb.Resize(image_size, image_size, p=1, always_apply=True)])
    dataset_valid = DatasetSegmentationBinary(transform_val)
    dataset_valid.load_from_json(path_json_val, path_dir_root)

    #model = UpashUnet_Light.load_from_checkpoint(path_model, in_channels=3, out_classes=1, loss_func=None)
    model = SmpModel_Light.load_from_checkpoint(path_model)
    model.eval()

    for i in range(0, 10):
        img, mask = dataset_valid.__getitem__(i)
        img = img.unsqueeze(0)
        print(img.size())
        with torch.no_grad():
            mask_pr = model(img)
            print(mask_pr.size())
            mask_pr = mask_pr[0][0].cpu().numpy() * 255

            cv2.imwrite("./lightning_out/m-" + str(i) + ".png", mask_pr)





path_json_train = "examples/data/datasets/seg-bin-train.json"
path_json_val = "examples/data/datasets/seg-bin-val.json"
path_dir_root = "/ml_auto_trainer/examples/data"
image_size = 512
in_channels = 3
out_channels = 1

#run_train(path_json_train, path_json_val, path_dir_root, image_size, None, None, in_channels, out_channels)

path_model = "./lightning_logs/version_3/checkpoints/epoch=1-step=334.ckpt"
run_test(path_json_val, path_dir_root, path_model)