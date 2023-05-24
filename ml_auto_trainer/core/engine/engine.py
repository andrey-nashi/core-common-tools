import pytorch_lightning as pl
from torch.utils.data import DataLoader


from .experiment import Experiment


class Engine:

    @staticmethod
    def run_trainer(exp: Experiment):
        if not exp.is_train_built: return

        train_dataloader = DataLoader(exp.dataset_train, batch_size=exp.engine_batch_size, shuffle=True, num_workers=exp.engine_threads)
        valid_dataloader = DataLoader(exp.dataset_valid, batch_size=exp.engine_batch_size, shuffle=False, num_workers=exp.engine_threads)

        trainer = pl.Trainer(accelerator=exp.engine_device, devices=1, max_epochs=exp.engine_epochs)
        trainer.fit(exp.model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)