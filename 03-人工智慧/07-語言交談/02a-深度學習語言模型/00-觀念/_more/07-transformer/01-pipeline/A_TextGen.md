# pipeline A

Text Generation

```
$ python pipelineA.py
2022-03-07 09:48:02.991236: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-07 09:48:02.991753: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
No model was supplied, defaulted to gpt2 (https://huggingface.co/gpt2)
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
[{'generated_text': 'As far as I am concerned, I will be the first to admit that I am not a fan of the idea of a "free market." I think that the idea of a free market is a bit of a stretch. I think that the idea'}]
```

原本句子只有 As far as I am concerned, I will ，但程式自動產生了後續的文章，變成

As far as I am concerned, I will be the first to admit that I am not a fan of the idea of a "free market." I think that the idea of a free market is a bit of a stretch. I think that the idea

因為 max_length=50 ，所以只產生 50 個詞 ...


