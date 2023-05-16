import torch

class BinaryCrossEntropyLoss(torch.nn.Module):

    def __init__(self, with_logits: bool = False):
        """
        Implements pytorch binary cross entropy (BCELoss) and binary cross
        entropy with logits (BCEWIthLogtisLoss).
        :param args:
        """
        super(BinaryCrossEntropyLoss, self).__init__()
        if not with_logits:
            self.criterion = torch.nn.BCELoss()
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Accepted tensors
        :param predictions: [B, class_count]
        :param targets: [B, class_count]
        :return:
        """

        # ---- For size missmatch as predictions [B,1,H,W] and [B,H,W] target
        if len(predictions.size()) == 4 and len(targets.size()) == 3 and predictions.size()[1] == 1:
            predictions = predictions[:, 0, :, :]

        if targets.type() != "torch.FloatTensor":
            targets = targets.float()
        loss = self.criterion(predictions, targets)
        return loss
