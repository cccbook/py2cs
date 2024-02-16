# F-Summarize

summarize

use Google’s T5 model. Even though it was pre-trained only on a multi-task mixed dataset (including CNN / Daily Mail), it yields very good results.

```
$ python F_Summarize.py
Downloading: 100%|██████████████████████████████████████████████████| 1.17k/1.17k [00:00<00:00, 599kB/s]
Downloading: 100%|███████████████████████████████████████████████████| 850M/850M [12:29<00:00, 1.19MB/s]
Downloading: 100%|████████████████████████████████████████████████████| 773k/773k [00:03<00:00, 246kB/s]
Downloading: 100%|██████████████████████████████████████████████████| 1.32M/1.32M [00:05<00:00, 270kB/s]
2022-03-07 11:04:46.159459: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-03-07 11:04:46.161274: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
<pad> prosecutors say the marriages were part of an immigration scam. if convicted, barrientos faces two 
criminal counts of "offering a false instrument for filing in the first degree" she has been married 10 times, nine of them between 1999 and 2002.</s>
```
