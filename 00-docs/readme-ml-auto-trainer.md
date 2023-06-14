**ml_auto_trainer**

---------------------------------------------------------
**MODELS**

| Model name      | Package             | Model args                                                                            |
|-----------------|---------------------|---------------------------------------------------------------------------------------|
| SmpModel_Light  | models.seg_smp      | model_name, encoder_name in_channels, out_classes, loss_func, is_save_log, activation |
| DenseNet_Light  | models.clf_densenet | model_name, out_classes, in_channels, is_trained, activation                          |

---------------------------------------------------------
**LOSSES**

| Loss name   | Package         | Args                |
|-------------|-----------------|---------------------|
| SmpDiceLoss | losses.smp_loss | mode, from_logits   |