import numpy as np
import torch
import pytorch_lightning as pl

from .model_raw import XLayerPerceptron

class XLayerPerceptron_Light(pl.LightningModule):

    DEFAULT_LAYER_CFG = XLayerPerceptron.DEFAULT_LAYER_CFG

    ACTIVATION_SIGMOID = XLayerPerceptron.ACTIVATION_SIGMOID
    ACTIVATION_SOFTMAX = XLayerPerceptron.ACTIVATION_SOFTMAX

    def __init__(self, dim_input: int, dim_output: int, loss_func: callable, lr: float=0.001, layer_cfg: list = DEFAULT_LAYER_CFG, activation: str = None):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_func"])

        self.model = XLayerPerceptron(dim_input, dim_output, layer_cfg, activation)
        self.loss_func = loss_func
        self.lr = lr

    def forward(self, features: torch.Tensor):
        return self.model(features)

    def shared_step(self, batch, batch_index, stage):
        features = batch[0]
        labels_gt = batch[1]

        labels_pr = self.forward(image)
        loss = self.loss_func(labels_pr, labels_gt)

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