import os
import shutil
from datetime import datetime
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .exp_obj import ExperimentTrain, ExperimentTest

class Engine:

    @staticmethod
    def run_trainer(exp: ExperimentTrain):
        if not exp.is_configured: return

        # ---- Prepare output directories
        if exp.engine_checkpoint is None:
            checkpoint_dir_path = "exp-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            checkpoint_dir_path = exp.engine_checkpoint
        if not os.path.exists(checkpoint_dir_path):
            os.makedirs(checkpoint_dir_path)
        else:
            shutil.rmtree(checkpoint_dir_path)
            os.makedirs(checkpoint_dir_path)

        train_dataloader = DataLoader(exp.dataset_train, batch_size=4, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(exp.dataset_valid, batch_size=4, shuffle=False, num_workers=4)

        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir_path, filename="model-weights", mode="min")

        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1, callbacks=[checkpoint_callback])
        trainer.fit(exp.model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    @staticmethod
    def run_tester(exp: ExperimentTest, exp_train):
        if not exp.is_configured: return

        test_dataloader = DataLoader(exp.dataset_test, batch_size=exp.engine_batch_size, shuffle=False, num_workers=1)

        exp.model.cuda()
        exp.model.eval()
        model = exp.model
        #<<<ERR
        #model = exp_train.model
        #model.cuda()
        #model.eval()

        with torch.no_grad():
            batch_id = 0
            progress_bar = tqdm(desc="Testing", total=len(test_dataloader))
            for batch in test_dataloader:
                progress_bar.update(1)
                sample_id = batch_id * exp.engine_batch_size
                batch_info = exp.dataset_test.get_data_source(sample_id, exp.engine_batch_size)
                batch_output = model(batch[0]).cpu().numpy()
                batch_input = [b.cpu().numpy() for b in batch]

                for p in exp.processors:
                    p.apply(batch_input, batch_output, batch_info)

                batch_id += 1

