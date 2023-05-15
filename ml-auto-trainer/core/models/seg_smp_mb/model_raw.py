import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SmpMultibranch(nn.Module):

    def __init__(self, model_name, in_channels_list: list, out_channels):
        self.encoder_list = []
        for in_channel in in_channels_list:
            encoder = smp.encoders.get_encoder(name="resnet34", in_channels=in_channel, depth=5, weights=None)
            self.encoder_list.append(encoder)

        decoder_channels = (256, 128, 64, 32, 16)


    def forward(self, image):
        return