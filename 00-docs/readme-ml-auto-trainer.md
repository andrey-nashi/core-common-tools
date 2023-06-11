**ml_auto_trainer**

---------------------------------------------------------
**MODELS**

| Model name       | Package        | Model args                                                                            |
|------------------|----------------|---------------------------------------------------------------------------------------|
| SmpModel_Light   | models.seg_smp | model_name, encoder_name in_channels, out_classes, loss_func, is_save_log, activation |


---------------------------------------------------------
**LOSSES**

| Loss name | Package          | Args |
|-----------|------------------|------|
| DiceLoss  | losses.dice_loss |      |