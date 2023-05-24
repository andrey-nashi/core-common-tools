import torch
from torch import nn

class DiceLoss(nn.Module):
     def __init__(self, smooth=1):
         """
         Implementation of DICE loss
         Reference: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
         :param smooth: smoothing parameter
         """
         super(DiceLoss, self).__init__()
         self.smooth = smooth
     def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
         """
         :param predictions: (B,H,W) or (B,C,H,W)
         :param targets: (B,H,W) or (B,C,H,W)
         """
         predictions = predictions.view(-1)
         targets = targets.view(-1)
         intersection = (predictions * targets).sum()
         dice = (2.0 * intersection + self.smooth) / (
             predictions.sum() + targets.sum() + self.smooth
         )
         return 1 - dice



