import torch
import torch.nn as nn

class FocalLoss(nn.Module):

    REDUCTION_MEAN = "mean"
    REDUCTION_SUM = "sum"

    def __init__(self, alpha: float = 0.25, gamma: int = 2, reduction: str = REDUCTION_MEAN):
        """
        Focal loss implementation with torch.autograd
        Reference: https://github.com/CoinCheung/pytorch-loss
        :param alpha:
        :param gamma:
        :param reduction:
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.criterion_bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param predictions: [B,C,W,H] or [B,W,H]
        :param targets: [B,C,W,H] or [B,W,H]
        """

        # ---- For size missmatch as predictions [B,1,H,W] and [B,H,W] target
        if len(predictions.size()) == 4 and len(targets.size()) == 3 and predictions.size()[1] == 1:
            predictions = predictions[:, 0, :, :]

        predictions = predictions.float()
        with torch.no_grad():
            alpha = torch.empty_like(predictions).fill_(1 - self.alpha)
            alpha[targets == 1] = self.alpha

        pt = torch.where(targets == 1, predictions, 1 - predictions)
        ce_loss = self.criterion_bce(predictions, targets.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == self.REDUCTION_MEAN:
            loss = loss.mean()
        if self.reduction == self.REDUCTION_SUM:
            loss = loss.sum()
        return loss

