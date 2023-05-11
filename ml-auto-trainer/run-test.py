

from core.models.seg_smp.model_pl import SmpModel_Light

path_model = "./lightning_logs/version_1/checkpoints/epoch=9-step=200.ckpt"

model = SmpModel_Light.load_from_checkpoint(path_model)