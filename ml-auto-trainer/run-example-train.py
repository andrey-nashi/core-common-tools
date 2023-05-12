import albumentations as alb
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from core.datasets.ds_segmentation import DatasetSegmentationBinary
from core.models.seg_smp.model_pl import SmpModel_Light


#-----------------------------------------------------------------------------------------
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
    train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(dataset_valid, batch_size=16, shuffle=False, num_workers=4)

    loss_func = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    model = SmpModel_Light(nn_architecture, nn_encoder, nn_in_channels, nn_out_channels, loss_func)

    trainer = pl.Trainer(max_epochs=30)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

#-----------------------------------------------------------------------------------------

if __name__ == '__main__':
    path_json_train = "/home/andrey/Dev/tote-data/td2-train.json"
    path_json_val = "/home/andrey/Dev/tote-data/td2-valid.json"
    path_dir_root = "/home/andrey/Dev/tote-data/"
    nn_image_size = 512
    nn_architecture = "Unet"
    nn_encoder = "resnet34"
    nn_in_channels = 3
    nn_out_channels = 1

    run_train(path_json_train, path_json_val, path_dir_root, nn_image_size, nn_architecture, nn_encoder, nn_in_channels, nn_out_channels)

