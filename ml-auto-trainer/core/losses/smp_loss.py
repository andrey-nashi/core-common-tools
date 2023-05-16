from segmentation_models_pytorch import losses


class SmpLoss:
    def __new__(cls, name, **kwargs):
        loss_cls = getattr(losses, name)
        loss = loss_cls(**kwargs)
        return loss
