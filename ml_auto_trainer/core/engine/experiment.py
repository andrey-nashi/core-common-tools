import os
import json

from ..losses import LossFactory
from ..models import ModelFactory
from ..datasets import DatasetFactory
from ..datasets import TransformationFactory


class ExperimentConfiguration:

    CFG_TRAIN = "train_cfg"
    CFG_TEST = "test_cfg"

    CFG_DS_TRAIN = "dataset_train"
    CFG_DS_VALID = "dataset_validation"
    CFG_DS_NAME = "name"
    CFG_DS_PATH_JSON = "path_json"
    CFG_DS_PATH_ROOT = "path_root"
    CFG_DS_ARGS = "args"

    CFG_TRANSFORM_TRAIN = "transform_train"
    CFG_TRANSFORM_VALID = "transform_valid"

    CFG_LOSS = "loss_func"
    CFG_LOSS_NAME = "name"
    CFG_LOSS_ARGS = "args"

    CFG_MODEL_NAME = "model_name"
    CFG_MODEL_ARGS = "model_args"

    CFG_ENGINE = "engine"
    CFG_ENGINE_BATCH_SIZE = "batch_size"
    CFG_ENGINE_EPOCHS = "epochs"
    CFG_ENGINE_THREADS = "threads"
    CFG_ENGINE_DEVICE = "device"

    @staticmethod
    def _exception_handler(func: callable):
        def _wrapper(self, *args, **kwargs):
            try:
                func(self, *args, **kwargs)
                return True
            except Exception:
                return False
        return _wrapper

    def __init__(self):
        self.is_train = False
        self.is_test = False

        self.ds_train_name = None
        self.ds_train_path_json = None
        self.ds_train_path_root = None
        self.ds_train_args = None

        self.ds_valid_name = None
        self.ds_valid_path_json = None
        self.ds_valid_path_root = None
        self.ds_valid_args = None

        self.transform_train = None
        self.transform_valid = None

        self.model_name = None
        self.model_args = None
        self.loss_name = None
        self.loss_args = None

        self.engine_batch_size = None
        self.engine_epochs = None
        self.engine_threads = None
        self.engine_device = None

    @_exception_handler
    def load_from_file(self, path_file: str):
        if not os.path.exists(path_file):
            raise FileNotFoundError

        f = open(path_file, "r")
        data = json.load(f)
        f.close()

        # ---- Training configuration
        if self.CFG_TRAIN in data:
            self.is_train = True
            cfg_train = data[self.CFG_TRAIN]

            # ---- Training dataset
            self.ds_train_name = cfg_train[self.CFG_DS_TRAIN][self.CFG_DS_NAME]
            self.ds_train_path_json = cfg_train[self.CFG_DS_TRAIN][self.CFG_DS_PATH_JSON]
            self.ds_train_path_root = cfg_train[self.CFG_DS_TRAIN][self.CFG_DS_PATH_ROOT]
            self.ds_train_args = cfg_train[self.CFG_DS_TRAIN][self.CFG_DS_ARGS]

            # ---- Validation dataset
            self.ds_valid_name = cfg_train[self.CFG_DS_VALID][self.CFG_DS_NAME]
            self.ds_valid_path_json = cfg_train[self.CFG_DS_VALID][self.CFG_DS_PATH_JSON]
            self.ds_valid_path_root = cfg_train[self.CFG_DS_VALID][self.CFG_DS_PATH_ROOT]
            self.ds_valid_args = cfg_train[self.CFG_DS_VALID][self.CFG_DS_ARGS]

            # ---- Transformation
            self.transform_train = cfg_train[self.CFG_TRANSFORM_TRAIN]
            self.transform_valid = cfg_train[self.CFG_TRANSFORM_VALID]

            # ---- Model
            self.model_name = cfg_train[self.CFG_MODEL_NAME]
            self.model_args = cfg_train[self.CFG_MODEL_ARGS]

            # ---- Loss
            self.loss_name = cfg_train[self.CFG_LOSS][self.CFG_LOSS_NAME]
            self.loss_args = cfg_train[self.CFG_LOSS][self.CFG_LOSS_ARGS]

            # ---- Engine
            self.engine_batch_size = cfg_train[self.CFG_ENGINE][self.CFG_ENGINE_BATCH_SIZE]
            self.engine_epochs = cfg_train[self.CFG_ENGINE][self.CFG_ENGINE_EPOCHS]
            self.engine_threads = cfg_train[self.CFG_ENGINE][self.CFG_ENGINE_THREADS]
            self.engine_device = cfg_train[self.CFG_ENGINE][self.CFG_ENGINE_DEVICE]

        # ---- Test configuration
        if self.CFG_TEST in data:
            self.is_test = True
            cfg_test = data[self.CFG_TEST]


class Experiment:

    def __init__(self):
        self.is_train_built = False
        self.dataset_train = None
        self.dataset_valid = None
        self.transform_train_func = None
        self.transform_valid_func = None
        self.model = None
        self.loss_func = None
        self.optimizer = None
        self.engine_batch_size = None
        self.engine_threads = None
        self.engine_epochs = None
        self.engine_device = None

    def build_train(self, exp_cfg: ExperimentConfiguration):
        if not exp_cfg.is_train: return

        # ---- Build transformation functions
        self.transform_train_func = TransformationFactory.create_transform(exp_cfg.transform_train)
        self.transform_valid_func = TransformationFactory.create_transform(exp_cfg.transform_valid)

        # ---- Build training dataset
        self.dataset_train = DatasetFactory.create_dataset(exp_cfg.ds_train_name, exp_cfg.ds_train_args)
        self.dataset_train.load_from_json(exp_cfg.ds_train_path_json, exp_cfg.ds_train_path_root)
        self.dataset_train.set_transform_func(self.transform_train_func)

        # ---- Build validation dataset
        self.dataset_valid = DatasetFactory.create_dataset(exp_cfg.ds_valid_name, exp_cfg.ds_valid_args)
        self.dataset_valid.load_from_json(exp_cfg.ds_valid_path_json, exp_cfg.ds_valid_path_root)
        self.dataset_valid.set_transform_func(self.transform_valid_func)

        # ---- Build model, loss function, optimizer
        self.loss_func = LossFactory.create_loss_function(exp_cfg.loss_name, exp_cfg.loss_args)

        self.model = ModelFactory.create_model(exp_cfg.model_name, exp_cfg.model_args)
        self.model.set_loss_func(self.loss_func)

        self.engine_batch_size = exp_cfg.engine_batch_size
        self.engine_threads = exp_cfg.engine_threads
        self.engine_epochs = exp_cfg.engine_epochs
        self.engine_device = exp_cfg.engine_device

        self.is_train_built = True

    def build_test(self, exp_cfg: ExperimentConfiguration):
        if not exp_cfg.is_test: return