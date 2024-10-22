

4.89 G ???

```
ccckmit@asus MINGW64 /d/ccc/ccc112b/py2cs/03-人工智慧/A3-HuggingFace/01-語言模型 (master)
$ python mistralai.py
tokenizer_config.json: 100%|█████████████████████████████████████████
████| 1.46k/1.46k [00:00<00:00, 1.46MB/s]
C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py:149: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\user\.cache\huggingface\hub\models--mistralai--Mixtral-8x7B-Instruct-v0.1. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to see activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
tokenizer.model: 100%|████████████████████████████████████████████
tokenizer.model: 100%|████████████████████████████████████████████
██████████| 493k/493k [00:00<00:00, 729kB/s]
tokenizer.json: 100%|█████████████████████████████████████████████
tokenizer.json: 100%|█████████████████████████████████████████████
███████| 1.80M/1.80M [00:00<00:00, 1.86MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████
█████████████| 72.0/72.0 [00:00<?, ?B/s]
config.json: 100%|██████████████████████████████████████████████
█████████████████████| 720/720 [00:00<?, ?B/s]
model.safetensors.index.json: 100%|██████████████████████████████████████
model.safetensors.index.json: 100%|██████████████████████████████████████
█| 92.7k/92.7k [00:00<00:00, 526kB/s]
Downloading shards:   0%|                                                              | 0/19 [00:00<?, ?it/s]
model-00001-of-00019.safetensors:   0%|▏                                 | 21.0M/4.89G [00:05<21:34, 3.76MB/s]
model-00001-of-00019.safetensors:   1%|▏                                 | 31.5M/4.89G [00:08<21:24, 3.78MB/s]
model-00001-of-00019.safetensors:   1%|▎                                 | 41.9M/4.89G [00:11<21:25, 3.77MB/s]
model-00001-of-00019.safetensors:   1%|▎                                 | 52.4M/4.89G [00:13<20:53, 3.86MB/s]
model-00001-of-00019.safetensors:   1%|▍                                 | 62.9M/4.89G [00:16<21:37, 3.72MB/s]
model-00001-of-00019.safetensors:   2%|▌                                 | 73.4M/4.89G [00:19<21:12, 3.79MB/s]
model-00001-of-00019.safetensors:   2%|▌                                 | 83.9M/4.89G [00:22<20:50, 3.84MB/s]
model-00001-of-00019.safetensors:   2%|▋                                 | 94.4M/4.89G [00:24<20:56, 3.82MB/s]
model-00001-of-00019.safetensors:   2%|▊                                  | 105M/4.89G [00:27<20:32, 3.88MB/s]
model-00001-of-00019.safetensors:   2%|▊                                  | 115M/4.89G [00:30<20:21, 3.91MB/s]
model-00001-of-00019.safetensors:   3%|▉                                  | 126M/4.89G [00:33<21:04, 3.77MB/s]
model-00001-of-00019.safetensors:   3%|▉                                  | 136M/4.89G [00:36<21:39, 3.66MB/s]
model-00001-of-00019.safetensors:   3%|█                                  | 147M/4.89G [00:40<25:01, 3.16MB/s]
model-00001-of-00019.safetensors:   3%|█▏                                 | 157M/4.89G [00:43<23:07, 3.41MB/s
model-00001-of-00019.safetensors:   3%|█▏                                 | 168M/4.89G [00:46<23:10, 3.40MB/s
model-00001-of-00019.safetensors:   4%|█▎                                 | 178M/4.89G [00:49<23:26, 3.35MB/s
model-00001-of-00019.safetensors:   4%|█▎                                 | 189M/4.89G [00:52<23:31, 3.33MB/s
model-00001-of-00019.safetensors:   4%|█▍                                 | 199M/4.89G [00:55<22:40, 3.45MB/s
model-00001-of-00019.safetensors:   4%|█▌                                 | 210M/4.89G [00:58<22:23, 3.49MB/s
model-00001-of-00019.safetensors:   5%|█▌                                 | 220M/4.89G [01:01<23:30, 3.31MB/s
model-00001-of-00019.safetensors:   5%|█▋                                 | 231M/4.89G [01:04<22:38, 3.43MB/s
model-00001-of-00019.safetensors:   5%|█▋                                 | 241M/4.89G [01:07<22:45, 3.41MB/s
model-00001-of-00019.safetensors:   5%|█▊                                 | 252M/4.89G [01:10<22:12, 3.48MB/s
model-00001-of-00019.safetensors:   5%|█▉                                 | 262M/4.89G [01:13<23:04, 3.35MB/s
model-00001-of-00019.safetensors:   6%|█▉                                 | 273M/4.89G [01:17<23:38, 3.26MB/s
model-00001-of-00019.safetensors:   6%|██                                 | 283M/4.89G [01:20<22:52, 3.36MB/s
model-00001-of-00019.safetensors:   6%|██                                 | 294M/4.89G [01:22<21:43, 3.53MB/s
model-00001-of-00019.safetensors:   6%|██▏                                | 304M/4.89G [01:26<22:06, 3.46MB/
model-00001-of-00019.safetensors:   6%|██▎                                | 315M/4.89G [01:28<21:21, 3.57MB/
model-00001-of-00019.safetensors:   7%|██▎                                | 325M/4.89G [01:31<21:50, 3.49MB/
model-00001-of-00019.safetensors:   7%|██▍                                | 336M/4.89G [01:35<22:05, 3.44MB/
model-00001-of-00019.safetensors:   7%|██▍                                | 346M/4.89G [01:38<22:01, 3.44MB/
model-00001-of-00019.safetensors:   7%|██▌                                | 357M/4.89G [01:41<22:11, 3.41MB/
model-00001-of-00019.safetensors:   8%|██▋                                | 367M/4.89G [01:44<22:48, 3.31MB/
model-00001-of-00019.safetensors:   8%|██▋                                | 377M/4.89G [01:48<23:33, 3.20MB/
model-00001-of-00019.safetensors:   8%|██▊                                | 388M/4.89G [01:51<23:06, 3.25MB/
model-00001-of-00019.safetensors:   8%|██▊                                | 398M/4.89G [01:54<22:53, 3.27MB/
model-00001-of-00019.safetensors:   8%|██▉                                | 409M/4.89G [01:57<22:19, 3.35MB/
model-00001-of-00019.safetensors:   9%|███                                | 419M/4.89G [02:01<23:21, 3.19MB/
model-00001-of-00019.safetensors:   9%|███                                | 430M/4.89G [02:05<24:54, 2.99MB/
model-00001-of-00019.safetensors:   9%|███▏                               | 440M/4.89G [02:09<27:05, 2.74MB
model-00001-of-00019.safetensors:   9%|███▏                               | 451M/4.89G [02:14<28:05, 2.64MB
model-00001-of-00019.safetensors:   9%|███▎                               | 461M/4.89G [02:17<25:55, 2.85MB
model-00001-of-00019.safetensors:  10%|███▍                               | 472M/4.89G [02:19<24:01, 3.07MB
model-00001-of-00019.safetensors:  10%|███▍                               | 482M/4.89G [02:22<22:26, 3.28MB
model-00001-of-00019.safetensors:  10%|███▌                               | 493M/4.89G [02:24<20:17, 3.61MB
model-00001-of-00019.safetensors:  10%|███▌                               | 503M/4.89G [02:26<18:46, 3.90MB
model-00001-of-00019.safetensors:  11%|███▋                               | 514M/4.89G [02:29<17:45, 4.11MB
model-00001-of-00019.safetensors:  11%|███▊                               | 524M/4.89G [02:31<16:50, 4.32MB
model-00001-of-00019.safetensors:  11%|███▊                               | 535M/4.89G [02:33<16:12, 4.48MB
model-00001-of-00019.safetensors:  11%|███▉                               | 545M/4.89G [02:35<16:02, 4.52MB
model-00001-of-00019.safetensors:  11%|███▉                               | 556M/4.89G [02:38<16:06, 4.49MB
model-00001-of-00019.safetensors:  12%|████                               | 566M/4.89G [02:40<16:07, 4.47MB
model-00001-of-00019.safetensors:  12%|████▏                              | 577M/4.89G [02:43<16:54, 4.25M
model-00001-of-00019.safetensors:  12%|████▏                              | 587M/4.89G [02:45<17:01, 4.21M
model-00001-of-00019.safetensors:  12%|████▎                              | 598M/4.89G [02:48<17:43, 4.04M
model-00001-of-00019.safetensors:  12%|████▎                              | 608M/4.89G [02:51<17:44, 4.02M
model-00001-of-00019.safetensors:  13%|████▍                              | 619M/4.89G [02:53<17:53, 3.98M
model-00001-of-00019.safetensors:  13%|████▌                              | 629M/4.89G [02:56<18:11, 3.91M
model-00001-of-00019.safetensors:  13%|████▌                              | 640M/4.89G [02:59<18:19, 3.87M
model-00001-of-00019.safetensors:  13%|████▋                              | 650M/4.89G [03:02<18:18, 3.86M
model-00001-of-00019.safetensors:  14%|████▋                              | 661M/4.89G [03:04<17:44, 3.97M
model-00001-of-00019.safetensors:  14%|████▊                              | 671M/4.89G [03:07<18:08, 3.88M
model-00001-of-00019.safetensors:  14%|████▉                              | 682M/4.89G [03:09<17:28, 4.02M
model-00001-of-00019.safetensors:  14%|████▉                              | 692M/4.89G [03:12<16:42, 4.19M
model-00001-of-00019.safetensors:  14%|█████                              | 703M/4.89G [03:14<16:23, 4.26M
model-00001-of-00019.safetensors:  15%|█████                              | 713M/4.89G [03:16<15:37, 4.46M
model-00001-of-00019.safetensors:  15%|█████▏                             | 724M/4.89G [03:18<15:13, 4.56
model-00001-of-00019.safetensors:  15%|█████▎                             | 734M/4.89G [03:21<15:02, 4.61
model-00001-of-00019.safetensors:  15%|█████▎                             | 744M/4.89G [03:23<14:32, 4.75
model-00001-of-00019.safetensors:  15%|█████▍                             | 755M/4.89G [03:25<14:17, 4.82
model-00001-of-00019.safetensors:  16%|█████▍                             | 765M/4.89G [03:27<14:01, 4.90
model-00001-of-00019.safetensors:  16%|█████▌                             | 776M/4.89G [03:29<13:57, 4.91
model-00001-of-00019.safetensors:  16%|█████▋                             | 786M/4.89G [03:31<13:57, 4.91
model-00001-of-00019.safetensors:  16%|█████▋                             | 797M/4.89G [03:33<13:40, 4.99
model-00001-of-00019.safetensors:  17%|█████▊                             | 807M/4.89G [03:36<14:30, 4.69
model-00001-of-00019.safetensors:  17%|█████▊                             | 818M/4.89G [03:39<16:03, 4.23
model-00001-of-00019.safetensors:  17%|█████▉                             | 828M/4.89G [03:41<16:23, 4.13
model-00001-of-00019.safetensors:  17%|██████                             | 839M/4.89G [03:44<16:20, 4.14
model-00001-of-00019.safetensors:  17%|██████                             | 849M/4.89G [03:47<16:49, 4.01
model-00001-of-00019.safetensors:  18%|██████▏                            | 860M/4.89G [03:50<17:27, 3.8
model-00001-of-00019.safetensors:  18%|██████▏                            | 870M/4.89G [03:52<17:07, 3.9
model-00001-of-00019.safetensors:  18%|██████▎                            | 881M/4.89G [03:55<16:38, 4.0
model-00001-of-00019.safetensors:  18%|██████▍                            | 891M/4.89G [03:57<16:24, 4.0
model-00001-of-00019.safetensors:  18%|██████▍                            | 902M/4.89G [04:00<16:39, 3.9
model-00001-of-00019.safetensors:  19%|██████▌                            | 912M/4.89G [04:02<16:38, 3.9
model-00001-of-00019.safetensors:  19%|██████▌                            | 923M/4.89G [04:05<16:14, 4.0
model-00001-of-00019.safetensors:  19%|██████▋                            | 933M/4.89G [04:07<15:49, 4.1
model-00001-of-00019.safetensors:  19%|██████▊                            | 944M/4.89G [04:10<16:32, 3.9
model-00001-of-00019.safetensors:  20%|██████▊                            | 954M/4.89G [04:13<15:49, 4.1
model-00001-of-00019.safetensors:  20%|██████▉                            | 965M/4.89G [04:15<15:09, 4.3
model-00001-of-00019.safetensors:  20%|██████▉                            | 975M/4.89G [04:17<14:54, 4.3
model-00001-of-00019.safetensors:  20%|███████                            | 986M/4.89G [04:19<14:46, 4.4
model-00001-of-00019.safetensors:  20%|███████▏                           | 996M/4.89G [04:22<15:36, 4.
model-00001-of-00019.safetensors:  21%|██████▉                           | 1.01G/4.89G [04:25<15:43, 4.1
model-00001-of-00019.safetensors:  21%|███████                           | 1.02G/4.89G [04:28<16:25, 3.9
model-00001-of-00019.safetensors:  21%|███████▏                          | 1.03G/4.89G [04:31<16:34, 3.
model-00001-of-00019.safetensors:  21%|███████▏                          | 1.04G/4.89G [04:33<16:57, 3.
model-00001-of-00019.safetensors:  21%|███████▎                          | 1.05G/4.89G [04:36<17:03, 3.
model-00001-of-00019.safetensors:  22%|███████▎                          | 1.06G/4.89G [04:39<16:50, 3.
model-00001-of-00019.safetensors:  22%|███████▍                          | 1.07G/4.89G [04:42<16:34, 3.
model-00001-of-00019.safetensors:  22%|███████▌                          | 1.08G/4.89G [04:45<16:55, 3.
model-00001-of-00019.safetensors:  22%|███████▌                          | 1.09G/4.89G [04:48<17:03, 3.
model-00001-of-00019.safetensors:  23%|███████▋                          | 1.10G/4.89G [04:52<16:47, 3.
76MB/s]
Downloading shards:   0%|                                                              | 0/19 [04:53<?, ?it/s]
Traceback (most recent call last):
  File "D:\ccc\ccc112b\py2cs\03-人工智慧\A3-HuggingFace\01-語言模型\mistralai.py", line 6, in <module>
    model = AutoModelForCausalLM.from_pretrained(model_id)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\models\auto\auto_factory.py", line 566, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\modeling_utils.py", line 3483, in from_pretrained
    resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\utils\hub.py", line 1025, in get_checkpoint_shard_files
    cached_filename = cached_file(
                      ^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\utils\hub.py", line 385, in cached_file
    resolved_file = hf_hub_download(
                    ^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\utils\_validators.py", line 118, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py", line 1457, in hf_hub_download
    http_get(
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py", line 524, in http_get
    for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\requests\models.py", line 816,
in generate
    yield from self.raw.stream(chunk_size, decode_content=True)
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\urllib3\response.py", line 628, in stream
    data = self.read(amt=amt, decode_content=decode_content)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\urllib3\response.py", line 567, in read
    data = self._fp_read(amt) if not fp_closed else b""
           ^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\urllib3\response.py", line 533, in _fp_read
    return self._fp.read(amt) if amt is not None else self._fp.read()
           ^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\http\client.py", line 466, in read
    s = self.fp.read(amt)
        ^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\socket.py", line 706, in readinto
    return self._sock.recv_into(b)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\ssl.py", line 1278, in recv_into
    return self.read(nbytes, buffer)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\ssl.py", line 1134, in read
    return self._sslobj.read(len, buffer)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
```