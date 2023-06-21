import numpy as np
import torch
import pytorch_lightning as pl

from .covidnet import CovidNet


class CovidNet_Light(pl.LightningModule):

    def __init__(self, in_channels: int = 3, out_classes: int = 1, model_type: str = CovidNet.TYPE_LARGE, activation: str = None, loss_func: callable = None):
        super().__init__()

        # ---- Force pytorch lighting to ignore saving loss function, and other flags
        self.save_hyperparameters(ignore=["loss_func"])

        self.model = CovidNet(num_classes=out_classes, in_channels=in_channels, model_type=model_type, activation=activation)

        self.loss_func = loss_func

        self.optimizer = None
        self.optimizer_lr = None

    def set_loss_func(self, loss_func: callable):
        self.loss_func = loss_func

    def set_optimizer(self, optimizer: callable, lr: float):
        self.optimizer = optimizer
        self.optimizer_lr = lr

    def forward(self, image: torch.Tensor):
        # ---- This check is useful for testing
        if image.device != self.device:
            image = image.to(self.device)
        label_vector = self.model.forward(image)

        return label_vector

    def training_step(self, batch, batch_index: int):
        image = batch[0]
        labels_gt = batch[1]

        assert image.ndim == 4
        assert image.max() <= 1.0 and image.min() >= -1

        labels_pr = self.forward(image)

        loss = self.loss_func(labels_pr, labels_gt)
        return loss

    def on_train_epoch_end(self):
        return

    def validation_step(self, batch, batch_index: int):
        image = batch[0]
        labels_gt = batch[1]

        assert image.ndim == 4
        assert image.max() <= 1.0 and image.min() >= -1

        labels_pr = self.forward(image)

        loss = self.loss_func(labels_pr, labels_gt)
        self.log('val_loss', loss)

    def on_validation_epoch_end(self):
        return

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), self.optimizer_lr)
        return optimizer

