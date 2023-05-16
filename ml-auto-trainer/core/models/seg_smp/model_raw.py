import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SmpModel(nn.Module):

    MODEL_UNET = "Unet"
    MODEL_UNETPP = "UnetPlusPlus"
    MODEL_MANET = "MAnet"
    MODEL_LINKNET = "Linknet"
    MODEL_FPN = "FPN"
    MODEL_PSPNET = "PSPNet"
    MODEL_DEEPLABV3 = "DeepLabV3"
    MODEL_DEEPLABV3P = "DeepLabV3Plus"
    MODEL_PAN = "PAN"

    _MODEL_TABLE = {
        MODEL_UNET: smp.Unet
    }


    MODEL_WEIGHTS_NONE = None
    MODEL_WEIGHTS_IMAGENET = "imagenet"

    MODEL_ACTIVATION_NONE = None
    MODEL_ACTIVATION_SIGMOID = "sigmoid"
    MODEL_ACTIVATION_SOFTMAX = "softmax"

    def __init__(self, model_name: str, encoder_name: str, in_channels: int, out_classes: int,
                 encoder_weights: str = MODEL_WEIGHTS_IMAGENET, activation: str = None):
        model_obj = self._MODEL_TABLE[model_name]
        self.model = model_obj(encoder_name=encoder_name, in_channels=in_channels, classes=out_classes,
                                encoder_weights=encoder_weights, activation=activation)


    def forward(self, x: torch.Tensor):
        return self.model(x)