import numpy as np
import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

class SmpModel_Light(pl.LightningModule):
    def __init__(self, smp_nn_model: str, encoder_name: str, in_channels: int, out_classes: int, loss_func: callable = None, is_save_log: bool = True, **kwargs):
        """
        Initialize segmentation model with given architecture, encoder, number of channels.
        :param smp_nn_model: model architecture [Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN]
        :param encoder_name: encoder for the given model. See https://github.com/qubvel/segmentation_models.pytorch for more details.
        Recommended encoders are
        * resnet18, resnet34, resnet50, resnet101, resnet152
        * efficientnet-b0 -> efficientnet-b7
        :param in_channels: number of channels, 3 for RGB input
        :param out_classes: number of output channels, 1 for binary segmentation
        :param loss_func: loss function (DICE is recommended)
        :param is_save_log: if True will save logs via PL hook, False, skip calculating and saving metrics
        :param kwargs:
        """
        super().__init__()

        # ---- Force pytorch lighting to ignore saving loss function, and other flags
        self.save_hyperparameters(ignore=["loss_func", "is_save_log"])

        # ---- Create smp model with given parameters
        self.model = smp.create_model(smp_nn_model, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes)

        # ---- Initialize imagenet like mean and standard deviation, will not work for non-3 channels _most_likely_
        # ---- FIXME add non-3 channel support
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_func = loss_func

        # ---- Cache is an internal variable for logging performance
        self.is_save_log = is_save_log
        self.cache = []

    def forward(self, image):
        # ---- This check is useful for testing
        if image.device != self.device:
            image = image.to(self.device)
        print(image.min(), image.max())
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        # <<<< CORE COMPUTE
        image = batch[0]
        assert image.ndim == 4

        mask_gt = batch[1]
        assert mask_gt.max() <= 1.0 and mask_gt.min() >= 0

        mask_pr = self.forward(image)
        loss = self.loss_func(mask_pr, mask_gt)
        # >>>> CORE COMPUTE

        # ---- Logging of various metrics to shared cache
        if stage == "valid" and self.is_save_log:
            prob_mask = mask_pr.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask_gt.long(), mode="binary")
            log_message = {
                "tp": tp.detach().cpu(),
                "fp": fp.detach().cpu(),
                "fn": fn.detach().cpu(),
                "tn": tn.detach().cpu()
            }
            self.cache.append(log_message)

        output = {"loss": loss}
        return output

    def shared_epoch_end(self, stage):

        # ---- Something went wrong
        if len(self.cache) == 0:
            return

        if self.is_save_log:
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
            # ---- Reset the cache for the next epoch
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
        return torch.optim.Adam(self.parameters(), lr=0.001)


    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict on a single image
        :param image: [H, W, 3] image UINT8
        :return: numpy array of the binary mask [H, W] mask(x,y) in [0|255]
        """
        transformed_image = torch.from_numpy(image / 255)
        transformed_image = transformed_image.permute(2, 0, 1)
        transformed_image= transformed_image.float()

        if self.training: self.eval()
        with torch.no_grad():
            model_output = self.forward(transformed_image)
            model_output = model_output[0][0].sigmoid().cpu().numpy()
            model_output = model_output * 255

        model_output = model_output.astype(np.uint8)
        print("-----------")
        print(model_output, ">>>>>>>>>>>>>>")
        return model_output