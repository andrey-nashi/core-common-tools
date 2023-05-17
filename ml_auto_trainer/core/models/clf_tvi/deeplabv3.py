import torch
import torch.nn as nn
import torchvision


class DeeplabV3(nn.Module):

    BACKBONE_RESNET50 = "resnet50"
    BACKBONE_RESNET101 = "resnet101"


    def __init__(self, num_classes: int=2, in_channels: int=3, is_trained: bool=True, backbone: str=BACKBONE_RESNET50):
        """
        Build the RESNEST50 network for classification.
        Input - [batch, in_channels, w, h]
        Output - [batch, num_classes, w, h]
        :param num_classes: (int) number of classes/output channels
        :param in_channels: (int) number of channels of the input tensor
        :param backbone: (str) backbone type: resnet50 or resnet101
        :param is_trained: (bool) flag, set to true to use pretrained
        """
        super(DeeplabV3, self).__init__()

        self.backbone = backbone
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.is_trained = is_trained

        if self.backbone == self.BACKBONE_RESNET50:
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=self.is_trained)
        elif self.backbone == self.BACKBONE_RESNET101:
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=self.is_trained)

        # ---- Initialize the initial layer
        if self.in_channels > 0 and self.in_channels != 3:
            self.model.backbone.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # ---- Initialize the final layer
        if self.num_classes != 21:
            self.model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = x["out"]
        return x
