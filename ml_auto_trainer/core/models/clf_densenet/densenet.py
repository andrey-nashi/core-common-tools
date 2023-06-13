import torch
import torch.nn as nn
import torchvision

# --------------------------------------------------------------------------------


class DenseNet121(nn.Module):

    ACTIVATION_SIGMOID = "sigmoid"
    ACTIVATION_SOFTMAX = "softmax"

    def __init__(self, num_classes: int = 1, in_channels: int = 3, is_trained: bool = True, activation: str = ACTIVATION_SIGMOID):
        """
        Build the DENSENET121 network for classification.
        Input - [batch, in_channels, w, h]
        Output - [batch, num_classes]
        :param num_classes: (int) number of classes/output channels
        :param is_trained: (bool) flag to specify whether to use pre-trained weights
        :param activation: (str) name of the activation function
        :param in_channels: (int) number of channels of the input tensor
        """
        super(DenseNet121, self).__init__()

        self.num_classes = num_classes
        self.is_trained = is_trained
        self.activation = activation
        self.in_channels = in_channels

        # ---- Build model, load trained weights from torchvision
        # weights = torchvision.models.DenseNet121_Weights.DEFAULT if self.is_trained else None
        # self.densenet121 = torchvision.models.densenet121(weights=weights)
        self.densenet121 = torchvision.models.densenet121(pretrained=self.is_trained)

        # ---- Build the fc layer with the specified activation function
        kernel_count = self.densenet121.classifier.in_features
        if self.activation == self.ACTIVATION_SIGMOID:
            self.densenet121.classifier = nn.Sequential(nn.Linear(kernel_count, self.num_classes), nn.Sigmoid())
        elif self.activation == self.ACTIVATION_SOFTMAX:
            self.densenet121.classifier = nn.Sequential(nn.Linear(kernel_count, self.num_classes), nn.Softmax(dim=1))
        elif self.activation is None:
            self.densenet121.classifier = nn.Sequential(nn.Linear(kernel_count, self.num_classes))

        # ---- Build the first layer with the specified number of input channels
        if self.in_channels > 0 and self.in_channels != 3:
            self.densenet121.features.conv0 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.densenet121(x)
        return x

# --------------------------------------------------------------------------------


class DenseNet169(nn.Module):

    ACTIVATION_SIGMOID = "sigmoid"
    ACTIVATION_SOFTMAX = "softmax"

    def __init__(self, num_classes: int = 1, in_channels: int = 3, is_trained: bool = True, activation: str = ACTIVATION_SIGMOID):
        """
        Build the DENSENET169 network for classification.
        Input - [batch, in_channels, w, h]
        Output - [batch, num_classes]
        :param num_classes: (int) number of classes/output channels
        :param is_trained: (bool) flag to specify whether to use pre-trained weights
        :param activation: (str) name of the activation function
        :param in_channels: (int) number of channels of the input tensor
        """
        super(DenseNet169, self).__init__()

        self.num_classes = num_classes
        self.is_trained = is_trained
        self.activation = activation
        self.in_channels = in_channels

        # ---- Build model, load trained weights from torchvision
        self.densenet169 = torchvision.models.densenet169(pretrained=self.is_trained)

        # ---- Build the fc layer with the specified activation function
        kernelCount = self.densenet169.classifier.in_features
        if self.activation == self.ACTIVATION_SIGMOID:
            self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, self.num_classes), nn.Sigmoid())
        elif self.activation == self.ACTIVATION_SOFTMAX:
            self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, self.num_classes), nn.Softmax())
        elif self.activation is None:
            self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, self.num_classes))

        # ---- Build the first layer with the specified number of input channels
        if self.in_channels > 0 and self.in_channels != 3:
            self.densenet169.features.conv0 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.densenet169(x)
        return x

# --------------------------------------------------------------------------------


class DenseNet201(nn.Module):

    ACTIVATION_SIGMOID = "sigmoid"
    ACTIVATION_SOFTMAX = "softmax"

    def __init__(self, num_classes: int = 1, in_channels: int = 3, is_trained: bool = True, activation: str = ACTIVATION_SIGMOID):
        """
        Build the DENSENET201 network for classification.
        Input - [batch, in_channels, w, h]
        Output - [batch, num_classes]
        :param num_classes: (int) number of classes/output channels
        :param is_trained: (bool) flag to specify whether to use pre-trained weights
        :param activation: (str) name of the activation function
        :param in_channels: (int) number of channels of the input tensor
        """

        super(DenseNet201, self).__init__()

        self.num_classes = num_classes
        self.is_trained = is_trained
        self.activation = activation
        self.in_channels = in_channels

        # ---- Build model, load trained weights from torchvision
        self.densenet201 = torchvision.models.densenet201(pretrained=self.is_trained)

        # ---- Build the fc layer with the specified activation function
        kernel_count = self.densenet201.classifier.in_features
        if self.activation == self.ACTIVATION_SIGMOID:
            self.densenet201.classifier = nn.Sequential(nn.Linear(kernel_count, self.num_classes), nn.Sigmoid())
        elif self.activation == self.ACTIVATION_SOFTMAX:
            self.densenet201.classifier = nn.Sequential(nn.Linear(kernel_count, self.num_classes), nn.Softmax())
        elif self.activation is None:
            self.densenet201.classifier = nn.Sequential(nn.Linear(kernel_count, self.num_classes))

        # ---- Build the first layer with the specified number of input channels
        if self.in_channels > 0 and self.in_channels != 3:
            self.densenet201.features.conv0 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.densenet201(x)
        return x
