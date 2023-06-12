from .exp_cfg import ExperimentConfiguration

from ..losses import LossFactory
from ..models import ModelFactory
from ..optimizer import OptimizerFactory
from ..datasets import DatasetFactory
from ..datasets import TransformationFactory
from ..processor import ProcessorFactory


class ExperimentTrain:

    def __init__(self):
        self.is_configured = False

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
        self.engine_checkpoint = None

    def configure(self, exp_cfg: ExperimentConfiguration):
        if not exp_cfg.is_train: return

        # ---- Build transformation functions
        self.transform_train_func = TransformationFactory.create_transform(exp_cfg.transform_train)
        self.transform_valid_func = TransformationFactory.create_transform(exp_cfg.transform_valid)

        # ---- Build training dataset
        self.dataset_train = DatasetFactory.create_dataset(exp_cfg.ds_train_name, exp_cfg.ds_train_args)
        is_ok = self.dataset_train.load_from_json(exp_cfg.ds_train_path_json, exp_cfg.ds_train_path_root)
        if not is_ok: raise FileNotFoundError
        self.dataset_train.set_transform_func(self.transform_train_func)

        # ---- Build validation dataset
        self.dataset_valid = DatasetFactory.create_dataset(exp_cfg.ds_valid_name, exp_cfg.ds_valid_args)
        is_ok = self.dataset_valid.load_from_json(exp_cfg.ds_valid_path_json, exp_cfg.ds_valid_path_root)
        if not is_ok: raise FileNotFoundError
        self.dataset_valid.set_transform_func(self.transform_valid_func)

        # ---- Build model, loss function, optimizer
        self.loss_func = LossFactory.create_loss_function(exp_cfg.loss_name, exp_cfg.loss_args)
        self.optimizer = OptimizerFactory.get_optimizer(exp_cfg.optimizer_name)

        self.model = ModelFactory.create_model(exp_cfg.model_name, exp_cfg.model_args)
        self.model.set_loss_func(self.loss_func)
        self.model.set_optimizer(self.optimizer, exp_cfg.optimizer_lr)

        self.engine_batch_size = exp_cfg.engine_batch_size
        self.engine_threads = exp_cfg.engine_threads
        self.engine_epochs = exp_cfg.engine_epochs
        self.engine_device = exp_cfg.engine_device
        self.engine_checkpoint = exp_cfg.engine_checkpoint

        self.is_configured = True


class ExperimentTest:

    def __init__(self):
        self.is_configured = False
        self.dataset_test = None
        self.transform_test_func = None
        self.model = None
        self.engine_batch_size = None
        self.processors = None

    def configure(self, exp_cfg: ExperimentConfiguration):
        if not exp_cfg.is_test: return

        self.transform_test_func = TransformationFactory.create_transform(exp_cfg.transform_test)

        self.dataset_test = DatasetFactory.create_dataset(exp_cfg.ds_test_name, exp_cfg.ds_test_args)
        is_ok = self.dataset_test.load_from_json(exp_cfg.ds_test_path_json, exp_cfg.ds_test_path_root)
        if not is_ok: raise FileNotFoundError
        self.dataset_test.set_transform_func(self.transform_test_func)

        self.model = ModelFactory.create_model(exp_cfg.model_name_t, exp_cfg.model_args_t)
        self.model.load_from_checkpoint(exp_cfg.model_checkpoint)

        self.engine_batch_size = exp_cfg.engine_batch_size_ts

        self.processors = []
        for proc_cfg in exp_cfg.processors:
            proc_name = proc_cfg["name"]
            proc_methods = proc_cfg["methods"]
            proc_args = proc_cfg["args"]
            p = ProcessorFactory.create_processor(proc_name, proc_methods, proc_args)
            self.processors.append(p)

        self.is_configured = True


class Experiment:

    def __init__(self):
        self.exp_cfg = None
        self.exp_train = ExperimentTrain()
        self.exp_test = ExperimentTest()

    def configure(self, exp_cfg: ExperimentConfiguration):
        self.exp_cfg = exp_cfg

    def get_train(self):
        self.exp_train.configure(self.exp_cfg)
        return self.exp_train

    def get_test(self):
        self.exp_test.configure(self.exp_cfg)
        return self.exp_test
