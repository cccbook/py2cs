# pipeline4

Here is an example of doing a sequence classification using a model to determine if two sequences are paraphrases of each other

兩句話互為釋義 (類似意義的語句, paraphrases of each other)

```
$ python pipeline3.py
2022-03-07 08:49:01.071645: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: 
cudart64_110.dll not found
2022-03-07 08:49:01.072219: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
Traceback (most recent call last):
  File "C:\ccc\course\ai\08-deep\07-transformer\01-pipeline\pipeline3.py", line 3, in <module>
    classifier = pipeline("sentiment-analysis")
  File "C:\Users\Hero3C\AppData\Local\Programs\Python\Python39\lib\site-packages\transformers\pipelines\__init__.py", line 598, in pipeline
    tokenizer = AutoTokenizer.from_pretrained(
  File "C:\Users\Hero3C\AppData\Local\Programs\Python\Python39\lib\site-packages\transformers\models\auto\tokenization_auto.py", line 546, in from_pretrained
    return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
  File "C:\Users\Hero3C\AppData\Local\Programs\Python\Python39\lib\site-packages\transformers\tokenization_utils_base.py", line 1724, in from_pretrained    resolved_vocab_files[file_id] = cached_path(
  File "C:\Users\Hero3C\AppData\Local\Programs\Python\Python39\lib\site-packages\transformers\file_utils.py", line 1921, in cached_path
    output_path = get_from_cache(
  File "C:\Users\Hero3C\AppData\Local\Programs\Python\Python39\lib\site-packages\transformers\file_utils.py", line 2177, in get_from_cache
    raise ValueError(
ValueError: Connection error, and we cannot find the requested files in the 
cached path. Please try again or make sure your Internet connection is on.  
$ python pipeline3.py
2022-03-07 08:49:40.752518: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: 
cudart64_110.dll not found
2022-03-07 08:49:40.753199: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
label: NEGATIVE, with score: 0.9991
label: POSITIVE, with score: 0.9999
$ python pipeline4.py
Downloading: 100%|██████████████████████| 29.0/29.0 [00:00<00:00, 2.04kB/s]
Downloading: 100%|█████████████████████████| 433/433 [00:00<00:00, 129kB/s]
Downloading: 100%|███████████████████████| 208k/208k [00:01<00:00, 140kB/s]
Downloading: 100%|███████████████████████| 426k/426k [00:02<00:00, 213kB/s]
Downloading: 100%|██████████████████████| 413M/413M [02:56<00:00, 2.46MB/s]
not paraphrase: 10%
is paraphrase: 90%
not paraphrase: 94%
is paraphrase: 6%
```
