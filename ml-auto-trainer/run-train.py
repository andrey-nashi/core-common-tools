import albumentations as alb
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from core.datasets.ds_segmentation import DatasetSegmentationBinary
from core.models.seg_smp.model_pl import SmpModel_Light

#-----------------------------------------------------------------------------------------
path_json_train = "/home/andrey/Dev/tote-data/td-train.json"
path_json_val = "/home/andrey/Dev/tote-data/td-val.json"
path_root = "/home/andrey/Dev/tote-data/"

transform_train = alb.Compose([
    alb.RandomRotate90(),
    alb.Flip(),
    alb.Transpose(),
    alb.IAAPerspective(),
    alb.OneOf([alb.IAAAdditiveGaussianNoise(), alb.GaussNoise()], p=0.2),
    alb.OneOf([alb.MotionBlur(p=0.2), alb.MedianBlur(blur_limit=3, p=0.1), alb.Blur(blur_limit=3, p=0.1)], p=0.2),
    alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    alb.OneOf([alb.OpticalDistortion(p=0.3), alb.GridDistortion(p=0.1), alb.IAAPiecewiseAffine(p=0.3)], p=0.2),
    alb.Resize(512, 512, p=1, always_apply=True),
])

transform_val = alb.Compose([alb.Resize(512, 512, p=1, always_apply=True)])


dataset_train = DatasetSegmentationBinary(transform_train)
dataset_train.serialize_from_json(path_json_train, path_root)
dataset_valid = DatasetSegmentationBinary(transform_val)
dataset_valid.serialize_from_json(path_json_val, path_root)
train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(dataset_valid, batch_size=16, shuffle=False, num_workers=4)

smp_architecture = "Unet"
smp_encoder = "resnet34"
in_channels = 3
out_channels = 1
loss_func = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
model = SmpModel_Light(smp_architecture, smp_encoder, in_channels, out_channels, loss_func)

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)