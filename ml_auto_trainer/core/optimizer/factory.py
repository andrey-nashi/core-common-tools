import torch

class OptimizerFactory:

    OPTIM_SGD = "SGD"
    OPTIM_ADAM = "ADAM"
    OPTIM_ADAMW = "ADAMW"
    OPTIM_ADAMAX = "ADAMAX"
    OPTIM_ADADELTA = "ADADELTA"
    OPTIM_ASGD = "ASGD"
    OPTIM_LBFGS = "LBFGS"
    OPTIM_RMSPROP = "RMSPROP"

    _TABLE_ = {
        OPTIM_SGD: torch.optim.SGD,
        OPTIM_ADAM: torch.optim.Adam,
        OPTIM_ADAMW: torch.optim.AdamW,
        OPTIM_ADAMAX: torch.optim.Adamax,
        OPTIM_ADADELTA: torch.optim.Adadelta,
        OPTIM_ASGD: torch.optim.ASGD,
        OPTIM_LBFGS: torch.optim.LBFGS,
        OPTIM_RMSPROP: torch.optim.RMSprop,
    }

    @staticmethod
    def get_optimizer(name: str):
        if name in OptimizerFactory._TABLE_:
            return OptimizerFactory._TABLE_[name]

