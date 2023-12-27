import torch
import torch.nn as nn


# --------------------------------------------------------------------

class Unit(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        """
        Basic convolution plus batch normalization unit
        :param in_channels: number of input channels in the unit
        :param out_channels: number of output channels in the unit
        """
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        output = self.bn(output)
        output = self.relu(output)

        return output


# --------------------------------------------------------------------


class ConvNet12(nn.Module):

    ACTIVATION_SIGMOID = "sigmoid"
    ACTIVATION_SOFTMAX = "softmax"

    def __init__(self, num_classes: int = 1, in_channels: int = 3, activation=ACTIVATION_SIGMOID):
        """
        Build the CONVNET12 network for classification.
        Input - [batch, in_channels, w, h]
        Output - [batch, num_classes]
        :param num_classes: (int) number of classes/output channels
        :param in_channels: (int) number of channels of the input tensor
        """
        super(ConvNet12, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.unit1 = Unit(in_channels=self.in_channels, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit3 = Unit(in_channels=32, out_channels=32)
        self.unit4 = Unit(in_channels=32, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit7 = Unit(in_channels=64, out_channels=128)
        self.unit8 = Unit(in_channels=128, out_channels=128)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.unit9 = Unit(in_channels=128, out_channels=256)
        self.unit10 = Unit(in_channels=256, out_channels=256)

        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.unit11 = Unit(in_channels=256, out_channels=512)
        self.unit12 = Unit(in_channels=512, out_channels=512)

        self.avgpool = nn.AdaptiveAvgPool2d((4,4))

        self.net = nn.Sequential(self.unit1, self.unit2, self.pool1, self.unit3, self.unit4, self.pool2, self.unit5,
                                 self.unit6, self.pool3, self.unit7, self.unit8, self.pool4, self.unit9, self.unit10,
                                 self.pool5, self.unit11, self.unit12, self.avgpool)

        self.fc = nn.Linear(in_features=512 * 16, out_features=self.num_classes)

        if activation == self.ACTIVATION_SIGMOID:
            self.activation = nn.Sigmoid()
        elif activation == self.ACTIVATION_SOFTMAX:
            self.activation = nn.Softmax()
        else:
            self.activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
