# pipeline B

text generation

* https://huggingface.co/blog/how-to-generate (讚)

Below is an example of text generation using XLNet and its tokenizer, which includes calling generate() directly

```
$ python pipelineB.py
Downloading: 100%|██████████████████████████████████████████████████████| 760/760 [00:00<00:00, 253kB/s]
Downloading: 100%|███████████████████████████████████████████████████| 445M/445M [03:14<00:00, 2.40MB/s]
Downloading: 100%|████████████████████████████████████████████████████| 779k/779k [00:03<00:00, 255kB/s]
Downloading: 100%|██████████████████████████████████████████████████| 1.32M/1.32M [00:05<00:00, 276kB/s]
2022-03-07 10:02:07.617997: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-07 10:02:07.618454: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Today the weather is really nice and I am planning on going out for a run and a walk tomorrow. It is an almost 90-degree day and I can't wait to get home and get back. The night before, when I have an appointment with the doctor, she asks me if I am ready to go... it's not very soon.... "Oh, yeah.
```
