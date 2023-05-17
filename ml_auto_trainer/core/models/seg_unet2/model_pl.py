import numpy as np
import torch
import pytorch_lightning as pl

from .model_raw import UpashUnet, UpashUnet2, UpashUnet3



class UpashUnet_Light(pl.LightningModule):

    def __init__(self, in_channels: int, out_classes: int, loss_func: callable, lr: float=0.001):
        super().__init__()
        self.model = UpashUnet(in_channels, out_classes)
        self.loss_func = loss_func
        self.lr = lr


    def forward(self, image: torch.Tensor):
        return self.model(image)


    def shared_step(self, batch, batch_index, stage):
        image = batch[0]
        mask_gt = batch[1]

        assert image.ndim == 4

        mask_pr = self.forward(image)
        loss = self.loss_func(mask_pr, mask_gt)

        return loss


    def training_step(self, batch, batch_index: int):
        return self.shared_step(batch, batch_index, "train")

    def on_train_epoch_end(self):
        #for some reason this hook is called AFTER valid epoch
        return

    def validation_step(self, batch, batch_index: int):
        return self.shared_step(batch, batch_index, "valid")

    def on_validation_epoch_end(self):
        return

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


