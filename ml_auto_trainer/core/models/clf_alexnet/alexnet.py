import torch
import torch.nn as nn
import torchvision


class AlexNet(nn.Module):

    ACTIVATION_SIGMOID = "sigmoid"
    ACTIVATION_SOFTMAX = "softmax"

    def __init__(self, num_classes: int = 1, is_trained: bool = True, activation: str = ACTIVATION_SIGMOID):
        """
        Build the ALEXNET network for classification.
        Input - [batch, in_channels, w, h]
        Output - [batch, num_classes]
        :param  num_classes: (int) number of classes/output channels
        :param is_trained: (bool) flag to specify whether to use pre-trained weights
        :param activation: (str) name of the activation function: SIGMOID, SOFTMAX, NONE
        """

        super(AlexNet, self).__init__()

        self.num_classes = num_classes
        self.is_trained = is_trained
        self.activation = activation

        if self.is_trained:
            self.alexnet = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)

        if self.activation == AlexNet.ACTIVATION_SIGMOID:
            self.alexnet.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, self.num_classes),
                nn.Sigmoid()
            )
        elif self.activation == AlexNet.ACTIVATION_SOFTMAX:
            self.alexnet.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, self.num_classes),
                nn.Softmax()
            )
        else:
            self.alexnet.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, self.num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alexnet(x)
        return x