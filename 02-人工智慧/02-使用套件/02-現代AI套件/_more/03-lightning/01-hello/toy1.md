# lightning

```
% python toy1.py
GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
/opt/miniconda3/lib/python3.12/site-packages/lightning/pytorch/loops/utilities.py:72: `max_epochs` was not set. Setting it to 1000 epochs. To train without an epoch limit, set `max_epochs=-1`.

  | Name  | Type   | Params | Mode 
-----------------------------------------
0 | model | Linear | 66     | train
-----------------------------------------
66        Trainable params
0         Non-trainable params
66        Total params
0.000     Total estimated model params size (MB)
1         Modules in train mode
0         Modules in eval mode
/opt/miniconda3/lib/python3.12/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.
/opt/miniconda3/lib/python3.12/site-packages/lightning/pytorch/loops/fit_loop.py:298: The number of training batches (8) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
Epoch 999: 100%|██| 8/8 [00:00<00:00, 950.79it/s, v_num=0]`Trainer.fit` stopped: `max_epochs=1000` reached.
Epoch 999: 100%|██| 8/8 [00:00<00:00, 771.38it/s, v_num=0]
```
