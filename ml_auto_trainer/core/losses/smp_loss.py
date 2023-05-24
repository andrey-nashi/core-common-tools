import torch
from torch import nn
import segmentation_models_pytorch as smp


class SmpDiceLoss(nn.Module):

    MODE_TABLE = {
        "binary": smp.losses.BINARY_MODE
    }

    def __init__(self, mode: str, from_logits: bool = True):
        super(SmpDiceLoss, self).__init__()
        smp_mode = self.MODE_TABLE[mode]
        self.criterion = smp.losses.DiceLoss(smp_mode, from_logits=from_logits)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions, targets)