from .dice_loss import DiceLoss
from .smp_loss import SmpDiceLoss
from .bce import BinaryCrossEntropyLoss

class LossFactory:

    _LOSSES = [
        DiceLoss,
        SmpDiceLoss,
        BinaryCrossEntropyLoss
    ]

    _TABLE_LOSSES = {m.__name__:m for m in _LOSSES}

    @staticmethod
    def create_loss_function(loss_name, loss_args):
        if loss_name in LossFactory._TABLE_LOSSES:
            return LossFactory._TABLE_LOSSES[loss_name](**loss_args)
        else:
            raise NotImplemented