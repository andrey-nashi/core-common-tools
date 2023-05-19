import os
import json


class ExperimentConfiguration:

    CFG_TRAIN = "train_cfg"
    CFG_TEST = "test_cfg"

    CFG_DS_TRAIN_NAME = "dataset_train"
    CFG_DS_TRAIN_ARGS = "dataset_train_args"

    CFG_DS_VALID_NAME = "dataset_valid"
    CFG_DS_VALID_ARGS = "dataset_valid_args"

    CFG_TRANSFORM_TRAIN = "transform_train"
    CFG_TRANSFORM_VALID = "transform_valid"

    CFG_MODEL_NAME = "model_name"
    CFG_MODEL_ARGS = "model_args"

    CFG_ENGINE = "engine"
    CFG_ENGINE_BATCH_SIZE = "batch_size"
    CFG_ENGINE_EPOCHS = "epochs"
    CFG_ENGINE_THREADS = "threads"
    CFG_ENGINE_DEVICE = "device"

    def __init__(self):
        self.dataset_train_name = None
        self.dataset_train_args = None

        self.dataset_valid_name = None
        self.dataset_valid_args = None

        self.transform_train = None
        self.transform_valid = None

        self.model_name = None
        self.model_args = None

        # ---- Training configuration
        self.engine_batch_size = None
        self.engine_epochs = None
        self.engine_threads = None
        self.engine_device = None

        self.is_train = False
        self.is_test = False

    def load(self, path_file: str):
        if not os.path.exists(path_file):
            raise FileNotFoundError

        f = open(path_file, "r")
        data = json.load(f)
        f.close()

        if self.CFG_TRAIN in data:
            self.is_train = True
            exp_train_cfg = data[self.CFG_TRAIN]
            self.dataset_train_name = exp_train_cfg[self.CFG_DS_TRAIN_NAME]
            self.dataset_train_args = exp_train_cfg[self.CFG_DS_TRAIN_ARGS]
            self.dataset_valid_name = exp_train_cfg[self.CFG_DS_VALID_NAME]
            self.dataset_valid_args = exp_train_cfg[self.CFG_DS_VALID_ARGS]
            self.transform_train = exp_train_cfg[self.CFG_TRANSFORM_TRAIN]
            self.transform_valid = exp_train_cfg[self.CFG_TRANSFORM_VALID]
            self.model_name = exp_train_cfg[self.CFG_MODEL_NAME]
            self.model_args = exp_train_cfg[self.CFG_MODEL_ARGS]

            engine_cfg = exp_train_cfg[self.CFG_ENGINE]
            self.engine_batch_size = engine_cfg[self.CFG_ENGINE_BATCH_SIZE]
            self.engine_epochs = engine_cfg[self.CFG_ENGINE_EPOCHS]
            self.engine_threads = engine_cfg[self.CFG_ENGINE_THREADS]
            self.engine_device = engine_cfg[self.CFG_ENGINE_DEVICE]
