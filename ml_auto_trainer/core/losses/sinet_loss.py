import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Sequence, Dict, Optional, Union

#------------------------------------------------------------
#---- Collection of losses for specific NN architectures.
#---- * SinetLoss - loss for SiNet
#---- * LossNN_OCNet - loss for OCNet
#------------------------------------------------------------

class SinetLoss(nn.Module):
    def __init__(self):
        """
        BCE based loss for the SiNet model for segmentation tasks.
        """
        super(SinetLoss, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param predictions: output of the SiNet model [BATCH, 2, H, W]
        :param targets: [BATCH, H, W], elements are class labels
        :return:
        """

        #---- SiNet produces 2 tensors, stored in a concatenated form
        out_sm = predictions[:, 0, :, :]
        out_im = predictions[:, 1, :, :]
        if targets.ndim == 4:
            targets = torch.squeeze(targets, dim=1)
        if targets.ndim != 3:
            raise ValueError("targets must be single channel masks. input shape is", targets.shape)
        loss = self.criterion(out_sm, targets.float()) + self.criterion(out_im, targets.float())
        return loss

class OcnetLoss(nn.Module):
    def __init__(self, dsn_weight: float = 0.4):
        """
        DSN loss for the OCNET based models
        :param dsn_weights: (float) weight argument in range (0,1)
        """
        super(OcnetLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.dsn_weight = dsn_weight

    def forward(self, predictions: Sequence[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:

        if targets.ndim == 4:
            targets = torch.squeeze(targets, dim=1)

        targets = targets.long()
        h, w = targets.size(1), targets.size(2)

        p1 = F.interpolate(input=predictions[0], size=(h, w), mode="bilinear", align_corners=True)
        x = torch.squeeze(targets, dim=1)
        loss1 = self.criterion(p1, x)

        p2 = F.interpolate(input=predictions[1], size=(h, w), mode="bilinear", align_corners=True)
        loss2 = self.criterion(p2, targets)

        loss = self.dsn_weight * loss1 + loss2
        return loss


