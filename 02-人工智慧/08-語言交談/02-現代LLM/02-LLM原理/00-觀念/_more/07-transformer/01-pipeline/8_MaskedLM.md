# pipeline8

Here is an example of doing masked language modeling using a model and a tokenizer.

```
$ python pipeline8.py
Downloading: 100%|███████████████████████████████████████████████████| 29.0/29.0 [00:00<00:00, 11.3kB/s]
Downloading: 100%|█████████████████████████████████████████████████████| 411/411 [00:00<00:00, 62.7kB/s]
Downloading: 100%|████████████████████████████████████████████████████| 208k/208k [00:00<00:00, 222kB/s]
Downloading: 100%|████████████████████████████████████████████████████| 426k/426k [00:02<00:00, 216kB/s]
Downloading: 100%|███████████████████████████████████████████████████| 251M/251M [01:49<00:00, 2.41MB/s]
2022-03-07 09:39:08.643263: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-07 09:39:08.643824: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help reduce our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help increase our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help decrease our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help offset our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help improve our carbon footprint.
```