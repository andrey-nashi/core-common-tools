import numpy as np
import torch
import pytorch_lightning as pl

from .model_raw import UNet, UNet2, UNet3



class UpashuUnet(pl.LightningModule):

    def __init__(self, in_channels, out_classes, loss_func):
        super().__init__()