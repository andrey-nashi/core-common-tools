import cv2
import torch
import albumentations as alb
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from core.datasets.ds_seg_bin import DatasetSegmentationBinary
from core.models.seg_smp.model_pl import SmpModel_Light

image_size = 512
path_json_train = "./examples/data/datasets/seg-bin-train.json"
path_dir_root = "./examples/data"
path_json_val = "./examples/data/datasets/seg-bin-val.json"

transform_train = alb.Compose([
    alb.RandomRotate90(),
    alb.Flip(),
    alb.Transpose(),
    alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
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

loss_f = smp.losses.DiceLoss("binary", from_logits=False)
#model = smp.Unet(encoder_name="resnet50", in_channels=3, classes=1, activation="sigmoid")
model = SmpModel_Light(model_name="Unet", encoder_name="resnet50", in_channels=3, out_classes=1, loss_func=loss_f, activation="sigmoid")
model.set_optimizer(torch.optim.SGD, lr=0.01)
#opt = torch.optim.SGD(model.parameters(),lr=0.01)

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
"""
model = model.cuda()
for epoch in range(0, 1):
    print("EPOCH", epoch)
    model.train()
    for batch in train_dataloader:
        opt.zero_grad()
        image = batch[0].cuda()
        mask_gt = batch[1].cuda()
        mask_pr = model(image)
        loss = loss_f(mask_pr, mask_gt)
        loss.backward()
        opt.step()

    model.eval()
    running_loss = 0
    index = 0
    with torch.no_grad():
        for batch in valid_dataloader:
            image = batch[0].cuda()
            mask_gt = batch[1].cuda()
            mask_pr = model(image)
            loss = loss_f(mask_pr, mask_gt)
            index += 1
            running_loss += loss

    print("val, los", running_loss /index)
"""
model.cuda()
model.eval()
with torch.no_grad():
    index_g = 0
    for batch in valid_dataloader:
        image = batch[0].cuda()
        mask_gt = batch[1].cuda()
        mask_pr = model(image)

        X = mask_pr.cpu().numpy()
        b, x, y, z = X.shape

        for bid in range(0, b):
            mask_out = X[bid][0] * 255
            cv2.imwrite("./lightning_out/exp-00x/img-" + str(index_g) + ".png", mask_out)

            index_g += 1


