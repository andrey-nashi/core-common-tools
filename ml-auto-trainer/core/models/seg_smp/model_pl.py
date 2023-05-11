import numpy as np
import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

class SmpModel_Light(pl.LightningModule):
    def __init__(self, smp_nn_model: str, encoder_name: str, in_channels: int, out_classes: int, loss_func: callable, is_log_iou: bool = True, **kwargs):
        super().__init__()

        self.save_hyperparameters(ignore=['loss_func'])

        self.model = smp.create_model(smp_nn_model, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs)

        # ---- Initialize imagenet like mean and standard deviation, will not work for non-3 channels
        # ---- FIXME add non-3 channel support
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_func = loss_func
        self.cache = []

    def forward(self, image):
        if image.device != self.device:
            image = image.to(self.device)
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]
        assert image.ndim == 4

        mask_gt = batch[1]
        assert mask_gt.max() <= 1.0 and mask_gt.min() >= 0

        mask_pr = self.forward(image)
        loss = self.loss_func(mask_pr, mask_gt)


        if stage == "valid":
            prob_mask = mask_pr.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask_gt.long(), mode="binary")
            self.cache.append({"tp": tp.cpu(), "fp": fp.cpu(), "fn": fn.cpu(), "tn": tn.cpu()})

        output = {"loss": loss}
        return output

    def shared_epoch_end(self, stage):

        if len(self.cache) == 0:
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
        self.cache = []

    def training_step(self, batch, batch_index: int):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        #for some reason this hook is called AFTER valid epoch
        return self.shared_epoch_end("train")

    def validation_step(self, batch, batch_index: int):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


    def predict(self, image: np.ndarray):
        transformed_image = torch.from_numpy(image)
        transformed_image = transformed_image.permute(2, 0, 1)

        if self.training: self.eval()
        with torch.no_grad():
            model_output = self(transformed_image)
            model_output = model_output[0][0].cpu().numpy()
            model_output = model_output * 255
        return model_output