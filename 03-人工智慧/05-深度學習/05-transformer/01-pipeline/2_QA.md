# pipeline2

question-answering

```
$ python pipeline1.py
2022-03-07 08:26:12.606638: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-07 08:26:12.607437: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
$ python pipeline2.py
2022-03-07 08:28:38.357888: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-07 08:28:38.358681: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
No model was supplied, defaulted to distilbert-base-cased-distilled-squad (https://huggingface.co/distilbert-base-cased-distilled-squad)
Downloading: 100%|█████| 473/473 [00:00<00:00, 73.9kB/s]
Downloading: 100%|███| 249M/249M [01:50<00:00, 2.36MB/s]
Downloading: 100%|███| 29.0/29.0 [00:00<00:00, 2.71kB/s]
Downloading: 100%|████| 208k/208k [00:01<00:00, 146kB/s]
Downloading: 100%|████| 426k/426k [00:01<00:00, 274kB/s]
{'score': 0.3097023665904999, 'start': 34, 'end': 58, 'answer': 'huggingface/transformers'}
```
