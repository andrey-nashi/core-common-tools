import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import albumentations as alb
from torch.utils.data import DataLoader
class DatasetSegmentation(Dataset):
    def __init__(self, transform):
        self.path_image_dir = None
        self.transform = transform
        self.table = []
        self.is_to_tensor = True

    def load_from_json(self, path_file, path_image_dir: str):
        f = open(path_file, "r")
        data = json.load(f)
        f.close()

        self.path_image_dir = path_image_dir

        for sample in data["dataset"]:
            path_image = sample["image"]
            path_mask = sample["mask"]

            self.table.append({"image": path_image, "mask": path_mask})

    def switch_flag_tt(self):
        self.is_to_tensor = not self.is_to_tensor

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        path_absolute_img = os.path.join(self.path_image_dir, self.table[idx]["image"])
        path_absolute_mask = os.path.join(self.path_image_dir, self.table[idx]["mask"])

        image = cv2.imread(path_absolute_img)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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


import random


class SmpModel_Light(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.cache = []

    def forward(self, image):
        print(self.mean)
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]

        assert image.ndim == 4

        mask = batch[1]

        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        output = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        if stage == "valid":
            self.cache.append({"tp": tp.cpu(),
                "fp": fp.cpu(),
                "fn": fn.cpu(),
                "tn": tn.cpu()})

        return output

    def shared_epoch_end(self, stage):
        # aggregate step metics
        if len(self.cache) == 0:
            print("!!!!EMPTY CACHE")
            return

        tp = torch.cat([x["tp"] for x in self.cache])
        fp = torch.cat([x["fp"] for x in self.cache])
        fn = torch.cat([x["fn"] for x in self.cache])
        tn = torch.cat([x["tn"] for x in self.cache])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)
        print(metrics)
        self.cache = []

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        #for some reason this hook is called AFTER valid epoch
        return self.shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


    def predict(self):
        return

#


"""
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
"""


def make_ds():
    ds_c_1 = []
    ds_c_2 = []



    def dsx(path_img, path_mask):
        out = []
        fl = [f for f in os.listdir(path_img) if f.endswith(".png")]
        for file in fl:
            if os.path.exists(os.path.join(path_mask, file)):
                path_img_r = os.path.join(os.path.basename(path_img), file)
                path_mask_r = os.path.join(os.path.basename(path_mask), file)
                out.append({"image": path_img_r, "mask": path_mask_r})
        return out

    path_img = "/home/andrey/Dev/tote-data/kojiya_220830_images"
    path_mask = "/home/andrey/Dev/tote-data/kojiya_220830_masks"
    ds_c_1 = dsx(path_img, path_mask)

    path_img = "/home/andrey/Dev/tote-data/kojiya_221229_images"
    path_mask = "/home/andrey/Dev/tote-data/kojiya_221229_masks"
    ds_c_2 = dsx(path_img, path_mask)

    random.shuffle(ds_c_1)
    random.shuffle(ds_c_2)

    l1 = int(0.85 * len(ds_c_1))
    l2 = int(0.85 * len(ds_c_2))

    ds_train = ds_c_1[0:l1] + ds_c_2[0:l2]
    ds_val = ds_c_1[l1:] + ds_c_2[l2:]

    f = open("/home/andrey/Dev/tote-data/td-train.json", "w")
    json.dump({"dataset": ds_train}, f, indent=4)
    f.close()

    f = open("/home/andrey/Dev/tote-data/td-val.json", "w")
    json.dump({"dataset": ds_val}, f, indent=4)
    f.close()

def trainX():
    f_train = "/home/andrey/Dev/tote-data/td-train.json"
    f_val = "/home/andrey/Dev/tote-data/td-val.json"
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


    d_train = DatasetSegmentation(transform_train)
    d_train.load_from_json(f_train, path_root)
    d_val = DatasetSegmentation(transform_val)
    d_val.load_from_json(f_val, path_root)
    train_dataloader = DataLoader(d_train, batch_size=16, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(d_val, batch_size=16, shuffle=False, num_workers=4)

    model = SmpModel_Light
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


def predict(model, img):


    h, w, c = img.shape
    print(img.shape)
    img = cv2.resize(img, (512, 512))
    transformed_image = torch.from_numpy(img)
    transformed_image = transformed_image.permute(2, 0, 1)
    transformed_image = transformed_image.cuda()
    print(transformed_image)
    model.eval()

    with torch.no_grad():
        x = model(transformed_image)
        x = x[0][0].cpu().numpy()

    x = cv2.resize(x, (w, h)) * 255
    return x



#trainX()


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



path_m = "./lightning_logs/version_1/checkpoints/epoch=9-step=200.ckpt"
model = SmpModel_Light.load_from_checkpoint(path_m)
path_in = "/home/andrey/Dev/tote-data/kojiya_220830_images"
path_out = "/home/andrey/Dev/tote-data/test-out"

img = cv2.imread("/home/andrey/Dev/test-in.jpg")
mask = predict(model, img)
oi = draw_mask(img, mask, (0, 0, 255), 0.5)
cv2.imwrite("/home/andrey/Dev/test-out.jpg", oi)
def rrr():
    fl =[f for f in os.listdir(path_in) if f.endswith(".png")]
    for file in fl:
        path_ff = os.path.join(path_in, file)
        path_fo = os.path.join(path_out, file)
        img = cv2.imread(path_ff)
        mask = predict(model, img)
        oi = draw_mask(img, mask, (0, 0, 255), 0.5)
        cv2.imwrite(path_fo, oi)

    path_in = "/home/andrey/Dev/tote-data/kojiya_221229_images"
    fl =[f for f in os.listdir(path_in) if f.endswith(".png")]
    for file in fl:
        path_ff = os.path.join(path_in, file)
        path_fo = os.path.join(path_out, file)
        img = cv2.imread(path_ff)
        mask = predict(model, img)
        oi = draw_mask(img, mask, (0, 0, 255), 0.5)
        h, w, c = oi.shape
        oi = cv2.resize(oi, (int(w/3), int(h/3)))
        cv2.imwrite(path_fo, oi)

