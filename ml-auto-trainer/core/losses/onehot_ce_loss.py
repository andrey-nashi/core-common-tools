import torch
from torch import nn


class OneHotCELoss(nn.Module):

    def __init__(self, dim=1, *args, **kwargs):
        """
        Implementation of a wrapper for cross entropy loss to support onehot encoding
        See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        """
        super().__init__()        
        self.dim = dim
        self.criterion = torch.nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = torch.argmax(targets, dim=self.dim)
        loss = self.criterion(predictions, targets)
        return loss