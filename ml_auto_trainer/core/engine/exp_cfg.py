import os
import yaml
import json

class ExperimentConfiguration:

    CFG_TRAIN = "train_cfg"
    CFG_TEST = "test_cfg"

    CFG_DS_TRAIN = "dataset_train"
    CFG_DS_VALID = "dataset_validation"
    CFG_DS_TEST = "dataset_test"
    CFG_DS_NAME = "name"
    CFG_DS_PATH_JSON = "path_json"
    CFG_DS_PATH_ROOT = "path_root"
    CFG_DS_ARGS = "args"

    CFG_TRANSFORM_TRAIN = "transform_train"
    CFG_TRANSFORM_VALID = "transform_valid"
    CFG_TRANSFORM_TEST = "transform_test"

    CFG_LOSS = "loss_func"
    CFG_LOSS_NAME = "name"
    CFG_LOSS_ARGS = "args"
    CFG_OPTIMIZER = "optimizer"
    CFG_OPTIM_NAME = "name"
    CFG_OPTIM_LR = "lr"

    CFG_MODEL_NAME = "model_name"
    CFG_MODEL_ARGS = "model_args"
    CFG_MODEL_CHECKPOINT = "model_checkpoint"

    CFG_ENGINE = "engine"
    CFG_ENGINE_BATCH_SIZE = "batch_size"
    CFG_ENGINE_EPOCHS = "epochs"
    CFG_ENGINE_THREADS = "threads"
    CFG_ENGINE_DEVICE = "device"
    CFG_ENGINE_CHECKPOINT = "checkpoint_path"

    CFG_PROCESSORS = "processors"

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
        self.cfg_source = None

        # ---- Training parameters
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
        self.optimizer_name = None
        self.optimizer_lr = None

        self.engine_batch_size = None
        self.engine_epochs = None
        self.engine_threads = None
        self.engine_device = None
        self.engine_checkpoint = None

        # ---- Testing parameters
        self.ds_test_name = None
        self.ds_test_path_json = None
        self.ds_test_path_root = None
        self.ds_test_args = None

        self.transform_test = None

        self.model_name_t = None
        self.model_args_t = None
        self.model_checkpoint = None

        self.engine_batch_size_ts = None

        self.processors = None

    @_exception_handler
    def load_from_file(self, path_file: str):
        if not os.path.exists(path_file):
            raise FileNotFoundError

        f = open(path_file, "r")
        data = json.load(f)
        f.close()

        self.cfg_source = data

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
            self.optimizer_name = cfg_train[self.CFG_OPTIMIZER][self.CFG_OPTIM_NAME]
            self.optimizer_lr = cfg_train[self.CFG_OPTIMIZER][self.CFG_OPTIM_LR]

            # ---- Engine
            self.engine_batch_size = cfg_train[self.CFG_ENGINE][self.CFG_ENGINE_BATCH_SIZE]
            self.engine_epochs = cfg_train[self.CFG_ENGINE][self.CFG_ENGINE_EPOCHS]
            self.engine_threads = cfg_train[self.CFG_ENGINE][self.CFG_ENGINE_THREADS]
            self.engine_device = cfg_train[self.CFG_ENGINE][self.CFG_ENGINE_DEVICE]
            self.engine_checkpoint = cfg_train[self.CFG_ENGINE][self.CFG_ENGINE_CHECKPOINT]

        # ---- Test configuration
        if self.CFG_TEST in data:
            self.is_test = True
            cfg_test = data[self.CFG_TEST]

            # --- Test dataset
            self.ds_test_name = cfg_test[self.CFG_DS_TEST][self.CFG_DS_NAME]
            self.ds_test_path_json = cfg_test[self.CFG_DS_TEST][self.CFG_DS_PATH_JSON]
            self.ds_test_path_root = cfg_test[self.CFG_DS_TEST][self.CFG_DS_PATH_ROOT]
            self.ds_test_args = cfg_test[self.CFG_DS_TEST][self.CFG_DS_ARGS]

            self.transform_test = cfg_test[self.CFG_TRANSFORM_TEST]

            # ---- Model
            self.model_name_t = cfg_test[self.CFG_MODEL_NAME]
            self.model_args_t = cfg_test[self.CFG_MODEL_ARGS]
            self.model_checkpoint = cfg_test[self.CFG_MODEL_CHECKPOINT]

            # ---- Batch based testing
            self.engine_batch_size_ts = cfg_test[self.CFG_ENGINE][self.CFG_ENGINE_BATCH_SIZE]

            # ---- Post-processing
            self.processors = cfg_test[self.CFG_PROCESSORS]
