from .smp_loss import SmpDiceLoss


class LossFactory:

    _LOSSES = [
        SmpDiceLoss
    ]

    _TABLE_LOSSES = {m.__name__:m for m in _LOSSES}

    @staticmethod
    def create_loss_function(loss_name, loss_args):
        if loss_name in LossFactory._TABLE_LOSSES:
            return LossFactory._TABLE_LOSSES[loss_name](**loss_args)
        else:
            raise NotImplemented