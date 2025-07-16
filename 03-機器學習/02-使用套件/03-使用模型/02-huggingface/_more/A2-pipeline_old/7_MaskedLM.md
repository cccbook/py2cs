# pipeline7

Masked Language Modeling

```
$ python pipeline7.py
2022-03-07 09:27:47.076782: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: 
cudart64_110.dll not found
2022-03-07 09:27:47.077518: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
No model was supplied, defaulted to distilroberta-base (https://huggingface.co/distilroberta-base)
Downloading: 100%|████████████████████████| 480/480 [00:00<00:00, 69.7kB/s]
Downloading: 100%|██████████████████████| 316M/316M [02:18<00:00, 2.39MB/s]
Downloading: 100%|███████████████████████| 878k/878k [00:03<00:00, 286kB/s]
Downloading: 100%|███████████████████████| 446k/446k [00:01<00:00, 231kB/s]
Downloading: 100%|█████████████████████| 1.29M/1.29M [00:04<00:00, 296kB/s]
[{'score': 0.17927542328834534,
  'sequence': 'HuggingFace is creating a tool that the community uses to solve '
              'NLP tasks.',
  'token': 3944,
  'token_str': ' tool'},
 {'score': 0.11349400132894516,
  'sequence': 'HuggingFace is creating a framework that the community uses to '
              'solve NLP tasks.',
  'token': 7208,
  'token_str': ' framework'},
 {'score': 0.05243544280529022,
  'sequence': 'HuggingFace is creating a library that the community uses to 
'
              'solve NLP tasks.',
  'token': 5560,
  'token_str': ' library'},
 {'score': 0.034935496747493744,
  'sequence': 'HuggingFace is creating a database that the community uses to '
              'solve NLP tasks.',
  'token': 8503,
  'token_str': ' database'},
 {'score': 0.02860252745449543,
  'sequence': 'HuggingFace is creating a prototype that the community uses to '
              'solve NLP tasks.',
  'token': 17715,
  'token_str': ' prototype'}]
```