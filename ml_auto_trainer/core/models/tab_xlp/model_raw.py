import torch
import torch.nn as nn


class XLayerPerceptron(nn.Module):

    DEFAULT_LAYER_CFG = [64, 32]

    ACTIVATION_SIGMOID = "sigmoid"
    ACTIVATION_SOFTMAX = "softmax"

    def __init__(self, dim_input: int, dim_output: int, layer_cfg: list = DEFAULT_LAYER_CFG, activation: str = None):
        super(XLayerPerceptron, self).__init__()
        assert len(layer_cfg) >= 2

        self.activation = activation
        self.layer_cfg = layer_cfg
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.layer_definition = []

        ls = [dim_input] + layer_cfg + [dim_output]
        for i in range(0, len(ls) - 1):
            fc = nn.Linear(ls[i], ls[i+1])
            self.layer_definition.append(fc)


    def forward(self, x):
        for index, fc in enumerate(self.layer_definition):
            x = fc(x)
            if index != len(self.layer_definition):
                x = torch.relu(x)

        if self.activation == self.ACTIVATION_SIGMOID:
            x = torch.sigmoid(x)
        elif self.activation == self.ACTIVATION_SOFTMAX:
            x = torch.softmax(x)

        return x