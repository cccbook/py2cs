
```sh
(py310) cccimac@cccimacdeiMac 01b-lightning % python linearRegressionPL.py
GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs

  | Name      | Type    | Params | Mode 
----------------------------------------------
0 | linear    | Linear  | 2      | train
1 | criterion | MSELoss | 0      | train
----------------------------------------------
2         Trainable params
0         Non-trainable params
2         Total params
0.000     Total estimated model params size (MB)
2         Modules in train mode
0         Modules in eval mode
/opt/homebrew/Caskroom/miniforge/base/envs/py310/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:425: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.
/opt/homebrew/Caskroom/miniforge/base/envs/py310/lib/python3.10/site-packages/pytorch_lightning/loops/fit_loop.py:310: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=100). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
Epoch 999: 100%|██████████████████████████████████████████| 1/1 [00:00<00:00, 416.85it/s, v_num=0]`Trainer.fit` stopped: `max_epochs=1000` reached.
Epoch 999: 100%|██████████████████████████████████████████| 1/1 [00:00<00:00, 257.81it/s, v_num=0]
Trained model: y = 1.0517x + 1.9152
2025-03-17 15:29:18.764 python[17007:94743] +[IMKClient subclass]: chose IMKClient_Legacy
2025-03-17 15:29:18.764 python[17007:94743] +[IMKInputSession subclass]: chose IMKInputSession_Legacy
```