# pipeline9

Causal Language Modeling

```
$ python pipeline9.py
Downloading: 100%|█████████████████████████████████████████████████████| 665/665 [00:00<00:00, 57.8kB/s]
Downloading: 100%|██████████████████████████████████████████████████| 0.99M/0.99M [00:03<00:00, 318kB/s]
Downloading: 100%|████████████████████████████████████████████████████| 446k/446k [00:01<00:00, 256kB/s]
Downloading: 100%|██████████████████████████████████████████████████| 1.29M/1.29M [00:04<00:00, 279kB/s]
Downloading: 100%|███████████████████████████████████████████████████| 523M/523M [03:43<00:00, 2.45MB/s]
2022-03-07 09:45:11.959403: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-07 09:45:11.960142: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Hugging Face is based in DUMBO, New York City, and directed
```

注意， directed 是預測的下一個最可能詞彙

一直預測下去，就能做文章產生器了 ...

