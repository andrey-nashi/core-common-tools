import os.path
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .experiment import Experiment

class Engine:

    @staticmethod
    def run_trainer(exp: Experiment):
        if not exp.is_built_train: return

        # ---- Prepare output directories
        if exp.engine_checkpoint is None:
            checkpoint_dir_path = "exp-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            checkpoint_dir_path = exp.engine_checkpoint
        if not os.path.exists(checkpoint_dir_path):
            os.makedirs(checkpoint_dir_path)

        train_dataloader = DataLoader(exp.dataset_train, batch_size=4, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(exp.dataset_valid, batch_size=4, shuffle=False, num_workers=4)

        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir_path, filename="model-weights", save_top_k=1, save_last=True)

        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1)
        trainer.fit(exp.model_train, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    @staticmethod
    def run_tester(exp: Experiment):
        if not exp.is_built_test: return

        test_dataloader = DataLoader(exp.dataset_test, batch_size=exp.engine_batch_size_ts, shuffle=False, num_workers=1)

        exp.model_train.cuda()
        exp.model_train.eval()

        with torch.no_grad():
            batch_id = 0
            for batch in test_dataloader:
                sample_id = batch_id * exp.engine_batch_size_ts
                batch_info = exp.dataset_test.get_data_source(sample_id, exp.engine_batch_size_ts)
                batch_output = exp.model_train(batch[0]).cpu().numpy()
                batch_input = [b.cpu().numpy() for b in batch]

                for p in exp.processors:
                    p.apply(batch_input, batch_output, batch_info)

                batch_id += 1

