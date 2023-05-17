import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = [w/sum(weights) for w in weights]

    def forward(self, predictions: dict, targets: dict) -> torch.Tensor:
        loss_results = [criterion(predictions, targets) for criterion in self.losses]
        loss = torch.stack(loss_results, dim=0).sum()
        return loss