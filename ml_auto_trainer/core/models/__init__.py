import inspect

from .clf_alexnet.model_pl import AlexNet_Light
from .clf_convnet.model_pl import ConvNet12_Light
from .clf_covidnet.model_pl import CovidNet_Light

from .clf_densenet.model_pl import DenseNet_Light

from .seg_smp.model_pl import SmpModel_Light
from .seg_deeplab.model_pl import DeeplabV3_Light




class ModelFactory:

    _LIST_MODEL_RAW = [
    ]

    _LIST_MODEL_PL = [
        AlexNet_Light,
        ConvNet12_Light,
        CovidNet_Light,
        DenseNet_Light,

        DeeplabV3_Light,
        SmpModel_Light
    ]

    _TABLE_MODEL_RAW = {m.__name__:m for m in _LIST_MODEL_RAW}
    _TABLE_MODEL_PL = {m.__name__:m for m in _LIST_MODEL_PL}

    @staticmethod
    def create_model(model_name, model_args):
        if model_name in ModelFactory._TABLE_MODEL_RAW:
            return ModelFactory._TABLE_MODEL_RAW[model_name](**model_args)
        elif model_name in ModelFactory._TABLE_MODEL_PL:
            return ModelFactory._TABLE_MODEL_PL[model_name](**model_args)
        else:
            raise NotImplemented
    @staticmethod
    def get_model_list():
        return [list(ModelFactory._TABLE_MODEL_RAW.keys()), list(ModelFactory._TABLE_MODEL_PL.keys())]

    @staticmethod
    def get_model_class(model_name):
        if model_name in ModelFactory._TABLE_MODEL_RAW:
            return ModelFactory._TABLE_MODEL_RAW[model_name]
        elif model_name in ModelFactory._TABLE_MODEL_PL:
            return ModelFactory._TABLE_MODEL_PL[model_name]
        else:
            raise NotImplemented

    @staticmethod
    def get_model_args(model_name: str):
        if model_name in ModelFactory._TABLE_MODEL_RAW:
            target_model = ModelFactory._TABLE_MODEL_RAW[model_name]
        elif model_name in ModelFactory._TABLE_MODEL_PL:
            target_model = ModelFactory._TABLE_MODEL_PL[model_name]
        else:
            raise NotImplemented

        output = {}
        model_args = inspect.signature(target_model.__init__).parameters
        for arg_name, arg in model_args.items():
            if arg_name != "self":
                output[arg_name] = {"ann": str(arg.annotation), "default": arg.default}

        return output