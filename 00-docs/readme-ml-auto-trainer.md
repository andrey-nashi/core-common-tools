**ml_auto_trainer**

---------------------------------------------------------
**MODELS**

| Model name      | Package             | Model args                                                                            |
|-----------------|---------------------|---------------------------------------------------------------------------------------|
| DenseNet_Light  | models.clf_densenet | model_name, out_classes, in_channels, is_trained, activation                          |
| AlexNet_Light   | models.clf_alexnet  | out_classes, is_trained, activation                                                   |
| ConvNet12_Light | models.clf_convnet  | in_channels, out_classes, activation                                                  |
| SmpModel_Light  | models.seg_smp      | model_name, encoder_name in_channels, out_classes, loss_func, is_save_log, activation |


---------------------------------------------------------
**LOSSES**

| Loss name              | Package         | Args              |
|------------------------|-----------------|-------------------|
| BinaryCrossEntropyLoss | losses.bce      | with_logits       |
| SmpDiceLoss            | losses.smp_loss | mode, from_logits |