# run

```
ccckmit@asus MINGW64 /d/ccc/ccc112b/py2cs/03-人工智慧/08-影像視覺/02-深度影像模型/04-diffusion/01-pretrained (master)
$ pip install --upgrade diffusers accelerate transformers
Collecting diffusers
  Downloading diffusers-0.26.3-py3-none-any.whl (1.9 MB)
     ---------------------------------------- 1.9/1.9 MB 1.5 MB/s eta 0:00:00
Collecting accelerate
  Downloading accelerate-0.27.2-py3-none-any.whl (279 kB)
     ---------------------------------------- 280.0/280.0 kB 1.7 MB/s eta 0:00:00
Requirement already satisfied: transformers in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (4.28.1)
Collecting transformers
  Downloading transformers-4.37.2-py3-none-any.whl (8.4 MB)
     ---------------------------------------- 8.4/8.4 MB 2.2 MB/s eta 0:00:00
Collecting importlib-metadata
  Downloading importlib_metadata-7.0.1-py3-none-any.whl (23 kB)
Requirement already satisfied: filelock in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from diffusers) (3.12.0)
Collecting huggingface-hub>=0.20.2
  Downloading huggingface_hub-0.20.3-py3-none-any.whl (330 kB)
     ---------------------------------------- 330.1/330.1 kB 1.7 MB/s eta 0:00:00
Requirement already satisfied: numpy in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from diffusers) (1.23.5)
Requirement already satisfied: regex!=2019.12.17 in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from diffusers) (2023.3.23)
Requirement already satisfied: requests in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from diffusers) (2.31.0)
Collecting safetensors>=0.3.1
  Downloading safetensors-0.4.2-cp311-none-win_amd64.whl (269 kB)
     ---------------------------------------- 269.6/269.6 kB 1.9 MB/s eta 0:00:00
Requirement already satisfied: Pillow in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from diffusers) (9.5.0)
Requirement already satisfied: packaging>=20.0 in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from accelerate) (23.1)
Requirement already satisfied: psutil in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from accelerate) (5.9.7)
Requirement already satisfied: pyyaml in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from accelerate) (6.0)
Requirement already satisfied: torch>=1.10.0 in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from accelerate) (2.0.1)
Collecting tokenizers<0.19,>=0.14
  Downloading tokenizers-0.15.2-cp311-none-win_amd64.whl (2.2 MB)
     ---------------------------------------- 2.2/2.2 MB 2.0 MB/s eta 0:00:00
Requirement already satisfied: tqdm>=4.27 in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from transformers) (4.65.0)
Collecting fsspec>=2023.5.0
  Downloading fsspec-2024.2.0-py3-none-any.whl (170 kB)
     ---------------------------------------- 170.9/170.9 kB 1.3 MB/s eta 0:00:00
Requirement already satisfied: typing-extensions>=3.7.4.3 in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from huggingface-hub>=0.20.2->diffusers) (4.8.0)
Requirement already satisfied: sympy in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from torch>=1.10.0->accelerate) (1.11.1)
Requirement already satisfied: networkx in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from torch>=1.10.0->accelerate) (3.1)
Requirement already satisfied: jinja2 in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from torch>=1.10.0->accelerate) (3.1.2)
Requirement already satisfied: colorama in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from tqdm>=4.27->transformers) (0.4.6)
Collecting zipp>=0.5
  Downloading zipp-3.17.0-py3-none-any.whl (7.4 kB)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from requests->diffusers) (3.1.0)
Requirement already satisfied: idna<4,>=2.5 in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from requests->diffusers) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from requests->diffusers) (1.26.15)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from requests->diffusers) (2022.12.7)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from jinja2->torch>=1.10.0->accelerate) (2.1.2)
Requirement already satisfied: mpmath>=0.19 in c:\users\user\appdata\local\programs\python\python311\lib\site-packages (from sympy->torch>=1.10.0->accelerate) (1.3.0)
Installing collected packages: zipp, safetensors, fsspec, importlib-metadata, huggingface-hub, tokenizers, diffusers, accelerate, transformers
  Attempting uninstall: fsspec
    Found existing installation: fsspec 2023.4.0
    Uninstalling fsspec-2023.4.0:
      Successfully uninstalled fsspec-2023.4.0
  Attempting uninstall: huggingface-hub
    Found existing installation: huggingface-hub 0.14.1
    Uninstalling huggingface-hub-0.14.1:
      Successfully uninstalled huggingface-hub-0.14.1
  Attempting uninstall: tokenizers
    Found existing installation: tokenizers 0.13.3
    Uninstalling tokenizers-0.13.3:
      Successfully uninstalled tokenizers-0.13.3
  Attempting uninstall: transformers
    Found existing installation: transformers 4.28.1
    Uninstalling transformers-4.28.1:
      Successfully uninstalled transformers-4.28.1
Successfully installed accelerate-0.27.2 diffusers-0.26.3 fsspec-2024.2.0 huggingface-hub-0.20.3 importlib-metadata-7.0.1 safetensors-0.4.2 tokenizers-0.15.2 transformers-4.37.2 zipp-3.17.0
WARNING: There was an error checking the latest version of pip.

ccckmit@asus MINGW64 /d/ccc/ccc112b/py2cs/03-人工智慧/08-影像視覺/02-深度影像模型/04-diffusion/01-pretrained (master)
$

ccckmit@asus MINGW64 /d/ccc/ccc112b/py2cs/03-人工智慧/08-影像視覺/02-深度影像模型/04-diffusion/01-pretrained (master)
$ python diffusionPretrained.py
model_index.json: 100%|████████████████████████████████████████████
███████████| 541/541 [00:00<00:00, 541kB/s]
C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py:149: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\user\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details,
see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to see activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
tokenizer/special_tokens_map.json: 100%|███████████████████████████████████
███| 472/472 [00:00<00:00, 471kB/s]%|                                            | 0.00/472 [00:00<?, ?B/s]

(…)ature_extractor/preprocessor_config.json: 100%|███████████████████████████| 342/
342 [00:00<00:00, 68.4kB/s]essor_config.json:   0%|                                  | 0.00/342 [00:00<?, ?B/s]text_encoder/config.json: 100%|████████████████████████████████████████
██████| 617/617 [00:00<00:00, 77.1kB/s]                                       | 0.00/617 [00:00<?, ?B/s]
scheduler/scheduler_config.json: 100%|████████████████████████████████████
███| 308/308 [00:00<00:00, 61.6kB/s]
safety_checker/config.json: 100%|███████████████████████████████████████
█| 4.72k/4.72k [00:00<00:00, 1.57MB/s]
Fetching 15 files:   7%|███▋                                                   | 1/15 [00:00<00:12,  1.12it
/s]
tokenizer/merges.txt: 100%|██████████████████████████████████████████
███████| 525k/525k [00:00<00:00, 807kB/s]                                   | 0.00/492M [00:00<?, ?B/s]
vae/config.json: 100%|████████████████████████████████████████████
███████████████████| 547/547 [00:00<?, ?B/s]                     | 0.00/547 [00:00<?, ?B/s]
unet/config.json: 100%|████████████████████████████████████████████
██████████████████| 743/743 [00:00<?, ?B/s]                       | 0.00/743 [00:00<?, ?B/s]
tokenizer/tokenizer_config.json: 100%|████████████████████████████████████
███████████| 806/806 [00:00<?, ?B/s]                                   | 0.00/1.06M [00:00<?, ?B/s]

tokenizer/vocab.json: 100%|██████████████████████████████████████████
█████| 1.06M/1.06M [00:02<00:00, 445kB/s]
diffusion_pytorch_model.safetensors:   0%|                                         | 0.00/335M [00:00<?, ?B/s]
model.safetensors:   0%|                                                          | 0.00/1.22G [00:00<?, ?B/s]
model.safetensors:   2%|█                                                 | 10.5M/492M [00:05<04:08, 1.94MB/s]
model.safetensors:   4%|██▏                                               | 21.0M/492M [00:11<04:10, 1.88MB/
model.safetensors:   6%|███▏                                              | 31.5M/492M [00:17<04:19, 1.78MB
model.safetensors:   9%|████▎                                             | 41.9M/492M [00:26<05:02, 1.49M
B/s]
model.safetensors:  11%|█████▎                                            | 52.4M/492M [00:35<05:23, 1.36
MB/s]sion_pytorch_model.safetensors:   0%|                              | 10.5M/3.44G [00:19<1:47:08, 533kB/s]
model.safetensors:  13%|██████▍                                           | 62.9M/492M [00:46<06:01, 1.1
9MB/s]
diffusion_pytorch_model.safetensors:   3%|█                                | 10.5M/335M [00:48<25:01, 216kB/s]

model.safetensors:  13%|██████▍                                           | 62.9M/492M [00:56<06:01, 1.1
model.safetensors:  15%|███████▌                                           | 73.4M/492M [01:02<07:30, 9
30kB/s]

diffusion_pytorch_model.safetensors:   3%|█                                | 10.5M/335M [01:05<25:01, 216kB/s]
diffusion_pytorch_model.safetensors:   2%|▍                             | 52.4M/3.44G [01:03<1:05:21, 863kB/s]
diffusion_pytorch_model.safetensors:   6%|██                               | 21.0M/335M [01:07<15:23, 340kB/s
model.safetensors:  15%|███████▌                                           | 73.4M/492M [01:16<07:30, 9
30kB/s]
diffusion_pytorch_model.safetensors:   2%|▌                               | 62.9M/3.44G [01:12<58:55, 955kB/s]
model.safetensors:   2%|▊                                               | 21.0M/1.22G [01:16<1:07:05, 297kB/s]
model.safetensors:  17%|████████▋                                          | 83.9M/492M [01:28<10:21,
657kB/s]n_pytorch_model.safetensors:   2%|▋                              | 73.4M/3.44G [01:19<52:03, 1.08MB/s]

diffusion_pytorch_model.safetensors:   9%|███                              | 31.5M/335M [01:29<12:59, 389kB/
s]ffusion_pytorch_model.safetensors:   2%|▊                              | 83.9M/3.44G [01:27<48:49, 1.14MB/s]
model.safetensors:  19%|█████████▊                                         | 94.4M/492M [01:45<10:24,
 637kB/s]
diffusion_pytorch_model.safetensors:   9%|███                              | 31.5M/335M [01:45<12:59, 389kB/
diffusion_pytorch_model.safetensors:  13%|████▏                            | 41.9M/335M [01:46<10:48, 452k
B/s]
model.safetensors:  19%|█████████▊                                         | 94.4M/492M [01:56<10:24,
 637kB/s]etensors:   3%|█▏                                              | 31.5M/1.22G [01:54<1:08:53, 287kB/s
]
model.safetensors:  21%|███████████                                         | 105M/492M [02:05<10:43
, 602kB/s]pytorch_model.safetensors:   3%|█                               | 115M/3.44G [02:01<55:03, 1.01MB/s]

model.safetensors:  23%|████████████▏                                       | 115M/492M [02:14<08:
58, 700kB/s]
diffusion_pytorch_model.safetensors:  16%|█████▏                           | 52.4M/335M [02:15<09:18, 506
model.safetensors:  23%|████████████▏                                       | 115M/492M [02:26<08:
model.safetensors:  26%|█████████████▎                                      | 126M/492M [02:27<08
:16, 738kB/s]orch_model.safetensors:   4%|█▎                              | 136M/3.44G [02:20<52:10, 1.05MB/s
]
model.safetensors:  28%|██████████████▍                                     | 136M/492M [02:40<0
7:54, 751kB/s]rch_model.safetensors:   4%|█▎                              | 147M/3.44G [02:27<46:37, 1.18MB/s
model.safetensors:  30%|███████████████▌                                    | 147M/492M [02:51<
07:06, 810kB/s]
diffusion_pytorch_model.safetensors:   5%|█▌                              | 168M/3.44G [02:42<42:57, 1.27MB/s
]
model.safetensors:   3%|█▋                                              | 41.9M/1.22G [02:54<1:25:16, 229kB/s
model.safetensors:  30%|███████████████▌                                    | 147M/492M [03:06<
07:06, 810kB/s]ch_model.safetensors:   5%|█▊                              | 189M/3.44G [02:58<42:06, 1.29MB/s
model.safetensors:  32%|████████████████▌                                   | 157M/492M [03:08
<07:36, 733kB/s]s:   3%|█▋                                              | 41.9M/1.22G [03:05<1:25:16, 229kB/s
]

model.safetensors:  34%|█████████████████▋                                  | 168M/492M [03:2
3<07:27, 725kB/s]


model.safetensors:  36%|██████████████████▊                                 | 178M/492M [03:
36<06:57, 752kB/s]


model.safetensors:   5%|██▌                                               | 62.9M/1.22G [03:35<51:03, 376kB/
model.safetensors:  36%|██████████████████▊                                 | 178M/492M [03:
model.safetensors:  38%|███████████████████▉                                | 189M/492M [03
:50<06:46, 747kB/s]
diffusion_pytorch_model.safetensors:  28%|█████████▎                       | 94.4M/335M [03:52<07:48,
 513kB/s]

model.safetensors:  38%|███████████████████▉                                | 189M/492M [04
:06<06:46, 747kB/s]odel.safetensors:   7%|██                             | 231M/3.44G [03:55<1:09:50, 765kB/s
]

diffusion_pytorch_model.safetensors:  34%|███████████▋                      | 115M/335M [04:11<05:0
diffusion_pytorch_model.safetensors:  38%|████████████▊                     | 126M/335M [04:18<04:
model.safetensors:  40%|█████████████████████                               | 199M/492M [0
4:22<08:59, 543kB/s]
diffusion_pytorch_model.safetensors:   7%|██▎                            | 252M/3.44G [04:18<1:03:49, 832kB/
model.safetensors:   6%|██▉                                             | 73.4M/1.22G [04:25<1:00:23, 315kB/
model.safetensors:  40%|█████████████████████                               | 199M/492M [0
4:36<08:59, 543kB/s]del.safetensors:   8%|██▌                              | 262M/3.44G [04:27<57:24, 922kB/
s]
diffusion_pytorch_model.safetensors:  44%|██████████████▉                   | 147M/335M [04:45<0
model.safetensors:  43%|██████████████████████▏                             | 210M/492M
[04:52<10:05, 466kB/s]
diffusion_pytorch_model.safetensors:  44%|██████████████▉                   | 147M/335M [04:55<0
3:54, 801kB/s]
model.safetensors:   7%|███▎                                            | 83.9M/1.22G [04:58<1:09:34, 271kB
model.safetensors:  45%|███████████████████████▎                            | 220M/492M
 [05:06<08:37, 526kB/s].safetensors:   9%|██▊                              | 294M/3.44G [04:58<55:27, 945kB/
s]
model.safetensors:  45%|███████████████████████▎                            | 220M/492M
 [05:16<08:37, 526kB/s].safetensors:   9%|██▉                              | 304M/3.44G [05:11<58:37, 891kB/
s]
model.safetensors:  47%|████████████████████████▎                           | 231M/492
M [05:22<07:50, 555kB/s]


diffusion_pytorch_model.safetensors:  53%|██████████████████                | 178M/335M [05:3
model.safetensors:  47%|████████████████████████▎                           | 231M/492
model.safetensors:  49%|█████████████████████████▍                          | 241M/49
2M [05:39<07:14, 577kB/s]afetensors:   9%|██▊                            | 315M/3.44G [05:35<1:00:18, 863kB/
s]
diffusion_pytorch_model.safetensors:  53%|██████████████████                | 178M/335M [05:4
5<03:47, 688kB/s]_model.safetensors:   9%|██▉                            | 325M/3.44G [05:42<1:08:23, 759kB/
model.safetensors:  51%|██████████████████████████▌                         | 252M/4
92M [05:53<06:32, 612kB/s]███▍                                              | 105M/1.22G [05:46<53:14, 348k
B/s]
diffusion_pytorch_model.safetensors:  56%|███████████████████▏              | 189M/335M [05
:57<04:04, 597kB/s]odel.safetensors:   9%|██▉                            | 325M/3.44G [05:55<1:08:23, 759kB/
model.safetensors:  51%|██████████████████████████▌                         | 252M/4
model.safetensors:  53%|███████████████████████████▋                        | 262M/
492M [06:06<05:48, 659kB/s]etensors:  10%|███                            | 336M/3.44G [06:04<1:20:38, 641kB/
s]
model.safetensors:  10%|█████▎                                             | 126M/1.22G [06:13<37:41, 482
model.safetensors:  55%|████████████████████████████▊                       | 273M
/492M [06:16<04:57, 739kB/s]tensors:  10%|███                            | 336M/3.44G [06:15<1:20:38, 641kB/
model.safetensors:  58%|█████████████████████████████▉                      | 283
M/492M [06:29<04:30, 774kB/s] ██▎                                             | 126M/1.22G [06:25<37:41, 482
kB/s]
diffusion_pytorch_model.safetensors:  60%|████████████████████▏             | 199M/335M [0
model.safetensors:  60%|███████████████████████████████                     | 29
4M/492M [06:39<03:56, 839kB/s]██▋                                             | 136M/1.22G [06:31<35:26, 508
kB/s]
diffusion_pytorch_model.safetensors:  60%|████████████████████▏             | 199M/335M [0
model.safetensors:  62%|████████████████████████████████                    | 3
04M/492M [06:48<03:27, 909kB/s] █▋                                             | 136M/1.22G [06:45<35:26, 508
kB/s]
model.safetensors:  64%|█████████████████████████████████▏                  |
 315M/492M [06:59<03:14, 914kB/s]
diffusion_pytorch_model.safetensors:  63%|█████████████████████▎            | 210M/335M [
07:04<05:07, 407kB/s]

model.safetensors:  66%|██████████████████████████████████▎
| 325M/492M [07:09<02:55, 954kB/s]█▌                                            | 157M/1.22G [07:08<32:19, 54
diffusion_pytorch_model.safetensors:  63%|█████████████████████▎            | 210M/335M [
model.safetensors:  68%|███████████████████████████████████▍
 | 336M/492M [07:20<02:46, 943kB/s]
diffusion_pytorch_model.safetensors:  11%|███▎                           | 367M/3.44G [07:15<1:41:45, 503kB
model.safetensors:  13%|██████▌                                            | 157M/1.22G [07:25<32:19, 54
diffusion_pytorch_model.safetensors:  66%|██████████████████████▎           | 220M/335M
model.safetensors:  70%|████████████████████████████████████▌
  | 346M/492M [07:34<02:43, 894kB/s]█                                            | 168M/1.22G [07:30<33:48, 51
7kB/s]

model.safetensors:  70%|████████████████████████████████████▌
model.safetensors:  72%|█████████████████████████████████████▋
   | 357M/492M [07:46<02:36, 870kB/s]

model.safetensors:  16%|███████▉                                           | 189M/1.22G [07:51<24:52, 6
model.safetensors:  75%|██████████████████████████████████████▊
    | 367M/492M [08:02<02:37, 794kB/s]█▎                                          | 199M/1.22G [08:01<21:50,
diffusion_pytorch_model.safetensors:  69%|███████████████████████▍          | 231M/335M
model.safetensors:  77%|███████████████████████████████████████▉
     | 377M/492M [08:14<02:21, 813kB/s]
diffusion_pytorch_model.safetensors:  69%|███████████████████████▍          | 231M/335M
 [08:15<04:36, 375kB/s]
model.safetensors:  16%|████████▎                                          | 199M/1.22G [08:15<21:50,
diffusion_pytorch_model.safetensors:  72%|████████████████████████▌         | 241M/335
model.safetensors:  77%|███████████████████████████████████████▉
     | 377M/492M [08:26<02:21, 813kB/s]
diffusion_pytorch_model.safetensors:  12%|███▌                           | 398M/3.44G [08:25<1:37:06, 522kB
/s]
diffusion_pytorch_model.safetensors:  72%|████████████████████████▌         | 241M/335
diffusion_pytorch_model.safetensors:  75%|█████████████████████████▌        | 252M/33
model.safetensors:  79%|████████████████████████████████████████▉
      | 388M/492M [08:41<02:49, 615kB/s]█▋                                         | 231M/1.22G [08:40<19:55,
 824kB/s]

diffusion_pytorch_model.safetensors:  12%|███▋                           | 409M/3.44G [08:45<1:34:38, 533kB
model.safetensors:  79%|████████████████████████████████████████▉
      | 388M/492M [08:56<02:49, 615kB/s]
model.safetensors:  81%|██████████████████████████████████████████
       | 398M/492M [09:05<02:50, 552kB/s] █▌                                        | 252M/1.22G [08:59<17:15
, 931kB/s]
diffusion_pytorch_model.safetensors:  78%|██████████████████████████▋       | 262M/3
diffusion_pytorch_model.safetensors:  81%|███████████████████████████▋      | 273M/
335M [09:08<01:53, 549kB/s]
model.safetensors:  81%|██████████████████████████████████████████
model.safetensors:  83%|███████████████████████████████████████████
▏        | 409M/492M [09:17<02:15, 613kB/s]██▉                           | 430M/3.44G [09:14<1:47:01, 468kB
diffusion_pytorch_model.safetensors:  85%|████████████████████████████▊     | 283M
/335M [09:21<01:24, 607kB/s]

model.safetensors:  85%|███████████████████████████████████████████4
█▎       | 419M/492M [09:33<01:55, 631kB/s]

model.safetensors:  85%|███████████████████████████████████████████0
█▎       | 419M/492M [09:46<01:55, 631kB/s]

model.safetensors:  22%|███████████▍                                       | 273M/1.22G [09:45<22:1
model.safetensors:  87%|███████████████████████████████████████████3
██▍      | 430M/492M [09:52<01:42, 610kB/s]█▉                           | 440M/3.44G [09:46<1:59:55, 417kB
diffusion_pytorch_model.safetensors: 100%|██████████████████████████████████|
 335M/335M [10:05<00:00, 552kB/s]
diffusion_pytorch_model.safetensors: 100%|█████████████████████████████████| 3
35M/335M [10:05<00:00, 1.01MB/s]

model.safetensors:  87%|███████████████████████████████████████████
██▍      | 430M/492M [10:06<01:42, 610kB/s]
model.safetensors:  89%|███████████████████████████████████████████B
███▌     | 440M/492M [10:13<01:31, 566kB/s]
diffusion_pytorch_model.safetensors:  13%|████                           | 451M/3.44G [10:10<1:58:07, 421kB
model.safetensors:  24%|████████████▎                                      | 294M/1.22G [10:13<24:
model.safetensors:  89%|███████████████████████████████████████████:
model.safetensors:  92%|███████████████████████████████████████████
████▋    | 451M/492M [10:26<01:06, 620kB/s]█                           | 451M/3.44G [10:25<1:58:07, 421kB
/s]
model.safetensors:  94%|███████████████████████████████████████████
█████▋   | 461M/492M [10:39<00:45, 675kB/s] ▏                                     | 315M/1.22G [10:33<18
:51, 797kB/s]

model.safetensors:  26%|█████████████▏                                     | 315M/1.22G [10:45<18
model.safetensors:  27%|█████████████▋                                     | 325M/1.22G [10:46<18
model.safetensors:  94%|███████████████████████████████████████████
model.safetensors:  96%|███████████████████████████████████████████6
██████▊  | 472M/492M [11:00<00:33, 608kB/s]


model.safetensors:  28%|██████████████▌                                    | 346M/1.22G [11:04<1
model.safetensors:  96%|███████████████████████████████████████████
model.safetensors:  98%|███████████████████████████████████████████1
███████▉ | 482M/492M [11:18<00:16, 596kB/s]
model.safetensors: 100%|███████████████████████████████████████████
█████████| 492M/492M [11:33<00:00, 710kB/s]█████████████████████████████<
█████████| 492M/492M [11:33<00:00, 621kB/s]
diffusion_pytorch_model.safetensors:  15%|████▌                          | 503M/3.44G [11:25<1:04:48, 755k
B/s]

model.safetensors:  31%|███████████████▌                                  | 377M/1.22G [11:32<1
3:02, 1.07MB/s]
model.safetensors:  32%|███████████████▉                                  | 388M/1.22G [11:38<1
model.safetensors:  33%|████████████████▍                                 | 398M/1.22G [11:44<
10:03, 1.36MB/s]
model.safetensors:  34%|████████████████▊                                 | 409M/1.22G [11:49<
09:00, 1.49MB/s]
model.safetensors:  34%|█████████████████▏                                | 419M/1.22G [11:54
model.safetensors:  35%|█████████████████▋                                | 430M/1.22G [12:00
<07:52, 1.66MB/s]
model.safetensors:  36%|██████████████████                                | 440M/1.22G [12:06
<07:22, 1.75MB/s]
model.safetensors:  37%|██████████████████▌                               | 451M/1.22G [12:1
3<07:38, 1.67MB/s]
model.safetensors:  38%|██████████████████▉                               | 461M/1.22G [12:2
0<08:08, 1.55MB/s]
model.safetensors:  39%|███████████████████▍                              | 472M/1.22G [12:
30<08:51, 1.40MB/s]
model.safetensors:  40%|███████████████████▊                              | 482M/1.22G [12:
39<09:22, 1.30MB/s]
diffusion_pytorch_model.safetensors:  17%|█████▎                          | 577M/3.44G [12:43<46:24, 1.03
model.safetensors:  41%|████████████████████▎                             | 493M/1.22G [12
:51<10:35, 1.14MB/s]
diffusion_pytorch_model.safetensors:  17%|█████▌                          | 598M/3.44G [12:56<38:13, 1.24
model.safetensors:  41%|████████████████████▋                             | 503M/1.22G [13
:04<11:49, 1.01MB/s]
model.safetensors:  42%|█████████████████████▏                            | 514M/1.22G [1
3:15<11:38, 1.01MB/s]
model.safetensors:  43%|█████████████████████▌                            | 524M/1.22G [1
model.safetensors:  44%|█████████████████████▉                            | 535M/1.22G [1
3:27<08:46, 1.29MB/s]
model.safetensors:  45%|██████████████████████▍                           | 545M/1.22G [
13:33<07:58, 1.40MB/s]
model.safetensors:  46%|██████████████████████▊                           | 556M/1.22G [
model.safetensors:  47%|███████████████████████▎                          | 566M/1.22G
[13:48<07:38, 1.42MB/s]
model.safetensors:  47%|███████████████████████▋                          | 577M/1.22G
model.safetensors:  48%|████████████████████████▏                         | 587M/1.22G
 [14:03<07:31, 1.39MB/s]
model.safetensors:  49%|████████████████████████▌                         | 598M/1.22G
 [14:11<07:41, 1.34MB/s]
model.safetensors:  50%|█████████████████████████                         | 608M/1.22G
model.safetensors:  51%|█████████████████████████▍                        | 619M/1.22
G [14:23<06:34, 1.51MB/s]
diffusion_pytorch_model.safetensors:  20%|██████▎                         | 682M/3.44G [14:25<44:53, 1.0
model.safetensors:  52%|█████████████████████████▊                        | 629M/1.22
model.safetensors:  53%|██████████████████████████▎                       | 640M/1.2
2G [14:36<05:55, 1.62MB/s]
model.safetensors:  53%|██████████████████████████▋                       | 650M/1.2
model.safetensors:  54%|███████████████████████████▏                      | 661M/1.
model.safetensors:  55%|███████████████████████████▌                      | 671M/1.
22G [14:52<05:02, 1.80MB/s]
model.safetensors:  56%|████████████████████████████                      | 682M/1.
22G [14:57<04:40, 1.91MB/s]
model.safetensors:  57%|████████████████████████████▍                     | 692M/1
model.safetensors:  58%|████████████████████████████▉                     | 703M/1
model.safetensors:  59%|█████████████████████████████▎                    | 713M/
1.22G [15:13<04:17, 1.95MB/s]
diffusion_pytorch_model.safetensors:  21%|██████▍                        | 713M/3.44G [15:15<1:03:26, 71
model.safetensors:  60%|█████████████████████████████▊                    | 724M/
model.safetensors:  60%|██████████████████████████████▏                   | 734M
/1.22G [15:25<04:23, 1.83MB/s]
model.safetensors:  61%|██████████████████████████████▌                   | 744M
model.safetensors:  62%|███████████████████████████████                   | 755M
/1.22G [15:39<04:41, 1.64MB/s]
model.safetensors:  63%|███████████████████████████████▍                  | 765
model.safetensors:  64%|███████████████████████████████▉                  | 776
M/1.22G [15:52<04:27, 1.64MB/s]
model.safetensors:  65%|████████████████████████████████▎                 | 78
6M/1.22G [15:59<04:23, 1.63MB/s]
model.safetensors:  66%|████████████████████████████████▊                 | 79
7M/1.22G [16:06<04:31, 1.54MB/s]
model.safetensors:  66%|█████████████████████████████████▏                | 8
07M/1.22G [16:14<04:34, 1.49MB/s]
model.safetensors:  67%|█████████████████████████████████▋                | 8
model.safetensors:  68%|██████████████████████████████████                | 8
28M/1.22G [16:28<04:19, 1.49MB/s]
model.safetensors:  69%|██████████████████████████████████▍               |
839M/1.22G [16:35<04:10, 1.50MB/s]
model.safetensors:  70%|██████████████████████████████████▉               |
model.safetensors:  71%|███████████████████████████████████▎              |
 860M/1.22G [16:48<03:49, 1.55MB/s]
model.safetensors:  72%|███████████████████████████████████▊              |
model.safetensors:  72%|████████████████████████████████████▏
| 881M/1.22G [17:01<03:31, 1.58MB/s]
model.safetensors:  73%|████████████████████████████████████▋
| 891M/1.22G [17:09<03:38, 1.48MB/s]
model.safetensors:  74%|█████████████████████████████████████
| 902M/1.22G [17:17<03:39, 1.43MB/s]
model.safetensors:  75%|█████████████████████████████████████▌
 | 912M/1.22G [17:24<03:28, 1.46MB/s]
model.safetensors:  76%|█████████████████████████████████████▉
 | 923M/1.22G [17:31<03:24, 1.43MB/s]
model.safetensors:  77%|██████████████████████████████████████▎
  | 933M/1.22G [17:40<03:26, 1.37MB/s]
diffusion_pytorch_model.safetensors:  25%|████████                        | 870M/3.44G [17:45<35:45, 1.
model.safetensors:  78%|██████████████████████████████████████▊
  | 944M/1.22G [17:51<03:51, 1.17MB/s]
diffusion_pytorch_model.safetensors:  26%|████████▎                       | 891M/3.44G [17:56<28:48, 1
model.safetensors:  78%|██████████████████████████████████████▊
model.safetensors:  78%|████████████████████████████████████████
    | 954M/1.22G [18:07<04:30, 967kB/s]
diffusion_pytorch_model.safetensors:  27%|████████▍                       | 912M/3.44G [18:08<25:56, 1
diffusion_pytorch_model.safetensors:  27%|████████▌                       | 923M/3.44G [18:13<25:05, 1
diffusion_pytorch_model.safetensors:  27%|████████▋                       | 933M/3.44G [18:19<23:44, 1
model.safetensors:  78%|████████████████████████████████████████
model.safetensors:  79%|████████████████████████████████████████▍
     | 965M/1.22G [18:28<05:31, 758kB/s]
diffusion_pytorch_model.safetensors:  28%|████████▉                       | 954M/3.44G [18:29<22:12, 1
diffusion_pytorch_model.safetensors:  28%|████████▉                       | 965M/3.44G [18:36<22:49, 1
model.safetensors:  80%|████████████████████████████████████████▉
     | 975M/1.22G [18:43<05:31, 726kB/s]
model.safetensors:  80%|████████████████████████████████████████▉
     | 975M/1.22G [18:55<05:31, 726kB/s]
model.safetensors:  81%|█████████████████████████████████████████▎
      | 986M/1.22G [18:57<05:13, 734kB/s]
model.safetensors:  82%|█████████████████████████████████████████▊
      | 996M/1.22G [19:08<04:36, 795kB/s]
model.safetensors:  83%|█████████████████████████████████████████▍
     | 1.01G/1.22G [19:17<03:59, 873kB/s]
diffusion_pytorch_model.safetensors:  30%|█████████▎                     | 1.03G/3.44G [19:17<26:42,
model.safetensors:  84%|█████████████████████████████████████████▊
     | 1.02G/1.22G [19:26<03:28, 954kB/s]
model.safetensors:  85%|██████████████████████████████████████████▎
      | 1.03G/1.22G [19:36<03:11, 982kB/s]
model.safetensors:  85%|█████████████████████████████████████████▊
model.safetensors:  86%|██████████████████████████████████████████▎
     | 1.05G/1.22G [19:50<02:22, 1.17MB/s]
diffusion_pytorch_model.safetensors:  31%|█████████▋                     | 1.07G/3.44G [19:51<31:14,
model.safetensors:  87%|██████████████████████████████████████████▋
     | 1.06G/1.22G [20:00<02:18, 1.13MB/s]
model.safetensors:  88%|███████████████████████████████████████████
     | 1.07G/1.22G [20:11<02:15, 1.08MB/s]
diffusion_pytorch_model.safetensors:  32%|█████████▉                     | 1.10G/3.44G [20:12<27:28,
diffusion_pytorch_model.safetensors:  32%|██████████                     | 1.11G/3.44G [20:18<25:55,
model.safetensors:  89%|███████████████████████████████████████████
█▍     | 1.08G/1.22G [20:25<02:21, 964kB/s]
model.safetensors:  89%|███████████████████████████████████████████,
█▍     | 1.08G/1.22G [20:35<02:21, 964kB/s]
model.safetensors:  90%|███████████████████████████████████████████,
█▊     | 1.09G/1.22G [20:38<02:18, 904kB/s]
model.safetensors:  91%|███████████████████████████████████████████,
██▎    | 1.10G/1.22G [20:46<01:55, 992kB/s]
model.safetensors:  91%|███████████████████████████████████████████,
model.safetensors:  92%|███████████████████████████████████████████
██▏   | 1.12G/1.22G [21:00<01:17, 1.21MB/s]
model.safetensors:  93%|███████████████████████████████████████████,
model.safetensors:  94%|███████████████████████████████████████████
███   | 1.14G/1.22G [21:14<00:53, 1.37MB/s]
model.safetensors:  95%|███████████████████████████████████████████,
███▍  | 1.15G/1.22G [21:20<00:43, 1.44MB/s]
model.safetensors:  96%|███████████████████████████████████████████,
███▉  | 1.16G/1.22G [21:26<00:34, 1.49MB/s]
model.safetensors:  97%|███████████████████████████████████████████3
████▎ | 1.17G/1.22G [21:34<00:29, 1.43MB/s]
model.safetensors:  97%|███████████████████████████████████████████5
model.safetensors: 100%|███████████████████████████████████████████
███████| 1.22G/1.22G [22:03<00:00, 919kB/s]██████████████████████████████,
model.safetensors:  99%|███████████████████████████████████████████
Fetching 15 files:  27%|█████████████▊                                      | 4/15 [22:05<1:05:39
, 358.16s/it] 1.22G/1.22G [22:03<00:00, 1.49MB/s]


diffusion_pytorch_model.safetensors:  36%|███████████▍                    | 1.23G/3.44G [22:02<39:3
3, 932kB/s]
diffusion_pytorch_model.safetensors:  36%|███████████▏                   | 1.24G/3.44G [22:08<33:15
diffusion_pytorch_model.safetensors:  36%|███████████▎                   | 1.25G/3.44G [22:12<27:31
diffusion_pytorch_model.safetensors:  37%|███████████▎                   | 1.26G/3.44G [22:16<23:42
diffusion_pytorch_model.safetensors:  37%|███████████▍                   | 1.27G/3.44G [22:20<20:54
diffusion_pytorch_model.safetensors:  37%|███████████▌                   | 1.28G/3.44G [22:24<18:49
diffusion_pytorch_model.safetensors:  38%|███████████▋                   | 1.29G/3.44G [22:29<17:25
diffusion_pytorch_model.safetensors:  38%|███████████▋                   | 1.30G/3.44G [22:33<16:24
diffusion_pytorch_model.safetensors:  38%|███████████▊                   | 1.31G/3.44G [22:37<15:40
diffusion_pytorch_model.safetensors:  38%|███████████▉                   | 1.32G/3.44G [22:42<15:24
diffusion_pytorch_model.safetensors:  39%|████████████                   | 1.33G/3.44G [22:46<14:57
diffusion_pytorch_model.safetensors:  39%|████████████                   | 1.34G/3.44G [22:50<14:35
diffusion_pytorch_model.safetensors:  39%|████████████▏                  | 1.35G/3.44G [22:54<14:0
diffusion_pytorch_model.safetensors:  40%|████████████▎                  | 1.36G/3.44G [22:58<13:5
diffusion_pytorch_model.safetensors:  40%|████████████▍                  | 1.37G/3.44G [23:02<13:5
diffusion_pytorch_model.safetensors:  40%|████████████▍                  | 1.38G/3.44G [23:06<13:3
diffusion_pytorch_model.safetensors:  41%|████████████▌                  | 1.39G/3.44G [23:11<14:3
diffusion_pytorch_model.safetensors:  41%|████████████▋                  | 1.41G/3.44G [23:16<14:3
diffusion_pytorch_model.safetensors:  41%|████████████▊                  | 1.42G/3.44G [23:21<14:3
diffusion_pytorch_model.safetensors:  41%|████████████▊                  | 1.43G/3.44G [23:25<13:5
diffusion_pytorch_model.safetensors:  42%|████████████▉                  | 1.44G/3.44G [23:29<13:4
diffusion_pytorch_model.safetensors:  42%|█████████████                  | 1.45G/3.44G [23:33<13:3
diffusion_pytorch_model.safetensors:  42%|█████████████▏                 | 1.46G/3.44G [23:38<13:
diffusion_pytorch_model.safetensors:  43%|█████████████▏                 | 1.47G/3.44G [23:42<13:
diffusion_pytorch_model.safetensors:  43%|█████████████▎                 | 1.48G/3.44G [23:46<13:
diffusion_pytorch_model.safetensors:  43%|█████████████▍                 | 1.49G/3.44G [23:51<13:
diffusion_pytorch_model.safetensors:  44%|█████████████▌                 | 1.50G/3.44G [23:55<13:
diffusion_pytorch_model.safetensors:  44%|█████████████▌                 | 1.51G/3.44G [24:00<13:
diffusion_pytorch_model.safetensors:  44%|█████████████▋                 | 1.52G/3.44G [24:04<13:
diffusion_pytorch_model.safetensors:  45%|█████████████▊                 | 1.53G/3.44G [24:08<13:
diffusion_pytorch_model.safetensors:  45%|█████████████▉                 | 1.54G/3.44G [24:13<13:
diffusion_pytorch_model.safetensors:  45%|█████████████▉                 | 1.55G/3.44G [24:17<13:
diffusion_pytorch_model.safetensors:  45%|██████████████                 | 1.56G/3.44G [24:22<13:
diffusion_pytorch_model.safetensors:  46%|██████████████▏                | 1.57G/3.44G [24:27<13
diffusion_pytorch_model.safetensors:  46%|██████████████▎                | 1.58G/3.44G [24:31<13
diffusion_pytorch_model.safetensors:  46%|██████████████▎                | 1.59G/3.44G [24:36<13
diffusion_pytorch_model.safetensors:  47%|██████████████▍                | 1.60G/3.44G [24:40<13
diffusion_pytorch_model.safetensors:  47%|██████████████▌                | 1.61G/3.44G [24:45<13
diffusion_pytorch_model.safetensors:  47%|██████████████▋                | 1.63G/3.44G [24:49<12
diffusion_pytorch_model.safetensors:  48%|██████████████▋                | 1.64G/3.44G [24:53<12
diffusion_pytorch_model.safetensors:  48%|██████████████▊                | 1.65G/3.44G [24:57<12
diffusion_pytorch_model.safetensors:  48%|██████████████▉                | 1.66G/3.44G [25:02<12
diffusion_pytorch_model.safetensors:  48%|███████████████                | 1.67G/3.44G [25:06<12
diffusion_pytorch_model.safetensors:  49%|███████████████▏               | 1.68G/3.44G [25:10<1
diffusion_pytorch_model.safetensors:  49%|███████████████▏               | 1.69G/3.44G [25:15<1
diffusion_pytorch_model.safetensors:  49%|███████████████▎               | 1.70G/3.44G [25:19<1
diffusion_pytorch_model.safetensors:  50%|███████████████▍               | 1.71G/3.44G [25:24<1
diffusion_pytorch_model.safetensors:  50%|███████████████▌               | 1.72G/3.44G [25:28<1
diffusion_pytorch_model.safetensors:  50%|███████████████▌               | 1.73G/3.44G [25:32<1
diffusion_pytorch_model.safetensors:  51%|███████████████▋               | 1.74G/3.44G [25:36<1
diffusion_pytorch_model.safetensors:  51%|███████████████▊               | 1.75G/3.44G [25:41<1
diffusion_pytorch_model.safetensors:  51%|███████████████▉               | 1.76G/3.44G [25:45<1
diffusion_pytorch_model.safetensors:  52%|███████████████▉               | 1.77G/3.44G [25:49<1
diffusion_pytorch_model.safetensors:  52%|████████████████               | 1.78G/3.44G [25:54<1
diffusion_pytorch_model.safetensors:  52%|████████████████▏              | 1.79G/3.44G [25:58<
diffusion_pytorch_model.safetensors:  52%|████████████████▎              | 1.80G/3.44G [26:02<
diffusion_pytorch_model.safetensors:  53%|████████████████▎              | 1.81G/3.44G [26:06<
diffusion_pytorch_model.safetensors:  53%|████████████████▍              | 1.82G/3.44G [26:10<
diffusion_pytorch_model.safetensors:  53%|████████████████▌              | 1.84G/3.44G [26:14<
diffusion_pytorch_model.safetensors:  54%|████████████████▋              | 1.85G/3.44G [26:19<
diffusion_pytorch_model.safetensors:  54%|████████████████▋              | 1.86G/3.44G [26:23<
diffusion_pytorch_model.safetensors:  54%|████████████████▊              | 1.87G/3.44G [26:27<
diffusion_pytorch_model.safetensors:  55%|████████████████▉              | 1.88G/3.44G [26:31<
diffusion_pytorch_model.safetensors:  55%|█████████████████              | 1.89G/3.44G [26:36<
diffusion_pytorch_model.safetensors:  55%|█████████████████              | 1.90G/3.44G [26:40<
diffusion_pytorch_model.safetensors:  56%|█████████████████▏             | 1.91G/3.44G [26:44
diffusion_pytorch_model.safetensors:  56%|█████████████████▎             | 1.92G/3.44G [26:48
diffusion_pytorch_model.safetensors:  56%|█████████████████▍             | 1.93G/3.44G [26:52
diffusion_pytorch_model.safetensors:  56%|█████████████████▍             | 1.94G/3.44G [26:57
diffusion_pytorch_model.safetensors:  57%|█████████████████▌             | 1.95G/3.44G [27:01
diffusion_pytorch_model.safetensors:  57%|█████████████████▋             | 1.96G/3.44G [27:06
diffusion_pytorch_model.safetensors:  57%|█████████████████▊             | 1.97G/3.44G [27:10
diffusion_pytorch_model.safetensors:  58%|█████████████████▊             | 1.98G/3.44G [27:16
diffusion_pytorch_model.safetensors:  58%|█████████████████▉             | 1.99G/3.44G [27:20
diffusion_pytorch_model.safetensors:  58%|██████████████████             | 2.00G/3.44G [27:25
diffusion_pytorch_model.safetensors:  59%|██████████████████▏            | 2.01G/3.44G [27:3
diffusion_pytorch_model.safetensors:  59%|██████████████████▏            | 2.02G/3.44G [27:3
diffusion_pytorch_model.safetensors:  59%|██████████████████▎            | 2.03G/3.44G [27:3
diffusion_pytorch_model.safetensors:  59%|██████████████████▍            | 2.04G/3.44G [27:4
diffusion_pytorch_model.safetensors:  60%|██████████████████▌            | 2.06G/3.44G [27:4
diffusion_pytorch_model.safetensors:  60%|██████████████████▋            | 2.07G/3.44G [27:5
diffusion_pytorch_model.safetensors:  60%|██████████████████▋            | 2.08G/3.44G [27:5
diffusion_pytorch_model.safetensors:  61%|██████████████████▊            | 2.09G/3.44G [28:0
diffusion_pytorch_model.safetensors:  61%|██████████████████▉            | 2.10G/3.44G [28:0
diffusion_pytorch_model.safetensors:  61%|███████████████████            | 2.11G/3.44G [28:0
diffusion_pytorch_model.safetensors:  62%|███████████████████            | 2.12G/3.44G [28:1
diffusion_pytorch_model.safetensors:  62%|███████████████████▏           | 2.13G/3.44G [28:
diffusion_pytorch_model.safetensors:  62%|███████████████████▎           | 2.14G/3.44G [28:
diffusion_pytorch_model.safetensors:  63%|███████████████████▍           | 2.15G/3.44G [28:
diffusion_pytorch_model.safetensors:  63%|███████████████████▍           | 2.16G/3.44G [28:
diffusion_pytorch_model.safetensors:  63%|███████████████████▌           | 2.17G/3.44G [28:
diffusion_pytorch_model.safetensors:  63%|███████████████████▋           | 2.18G/3.44G [28:
diffusion_pytorch_model.safetensors:  64%|███████████████████▊           | 2.19G/3.44G [28:
diffusion_pytorch_model.safetensors:  64%|███████████████████▊           | 2.20G/3.44G [28:
diffusion_pytorch_model.safetensors:  64%|███████████████████▉           | 2.21G/3.44G [28:
diffusion_pytorch_model.safetensors:  65%|████████████████████           | 2.22G/3.44G [28:
diffusion_pytorch_model.safetensors:  65%|████████████████████▏          | 2.23G/3.44G [29
diffusion_pytorch_model.safetensors:  65%|████████████████████▏          | 2.24G/3.44G [29
diffusion_pytorch_model.safetensors:  66%|████████████████████▎          | 2.25G/3.44G [29
diffusion_pytorch_model.safetensors:  66%|████████████████████▍          | 2.26G/3.44G [29
diffusion_pytorch_model.safetensors:  66%|████████████████████▌          | 2.28G/3.44G [29
diffusion_pytorch_model.safetensors:  66%|████████████████████▌          | 2.29G/3.44G [29
diffusion_pytorch_model.safetensors:  67%|████████████████████▋          | 2.30G/3.44G [29
diffusion_pytorch_model.safetensors:  67%|████████████████████▊          | 2.31G/3.44G [29
diffusion_pytorch_model.safetensors:  67%|████████████████████▉          | 2.32G/3.44G [29
diffusion_pytorch_model.safetensors:  68%|████████████████████▉          | 2.33G/3.44G [29
diffusion_pytorch_model.safetensors:  68%|█████████████████████          | 2.34G/3.44G [29
diffusion_pytorch_model.safetensors:  68%|█████████████████████▏         | 2.35G/3.44G [2
diffusion_pytorch_model.safetensors:  69%|█████████████████████▎         | 2.36G/3.44G [2
diffusion_pytorch_model.safetensors:  69%|█████████████████████▎         | 2.37G/3.44G [2
diffusion_pytorch_model.safetensors:  69%|█████████████████████▍         | 2.38G/3.44G [3
diffusion_pytorch_model.safetensors:  70%|█████████████████████▌         | 2.39G/3.44G [3
diffusion_pytorch_model.safetensors:  70%|█████████████████████▋         | 2.40G/3.44G [3
diffusion_pytorch_model.safetensors:  70%|█████████████████████▋         | 2.41G/3.44G [3
diffusion_pytorch_model.safetensors:  70%|█████████████████████▊         | 2.42G/3.44G [3
diffusion_pytorch_model.safetensors:  71%|█████████████████████▉         | 2.43G/3.44G [3
diffusion_pytorch_model.safetensors:  71%|██████████████████████         | 2.44G/3.44G [3
diffusion_pytorch_model.safetensors:  71%|██████████████████████         | 2.45G/3.44G [3
diffusion_pytorch_model.safetensors:  72%|██████████████████████▏        | 2.46G/3.44G [
diffusion_pytorch_model.safetensors:  72%|██████████████████████▎        | 2.47G/3.44G [
diffusion_pytorch_model.safetensors:  72%|██████████████████████▍        | 2.49G/3.44G [
diffusion_pytorch_model.safetensors:  73%|██████████████████████▌        | 2.50G/3.44G [
diffusion_pytorch_model.safetensors:  73%|██████████████████████▌        | 2.51G/3.44G [
diffusion_pytorch_model.safetensors:  73%|██████████████████████▋        | 2.52G/3.44G [
diffusion_pytorch_model.safetensors:  74%|██████████████████████▊        | 2.53G/3.44G [
diffusion_pytorch_model.safetensors:  74%|██████████████████████▉        | 2.54G/3.44G [
diffusion_pytorch_model.safetensors:  74%|██████████████████████▉        | 2.55G/3.44G [
diffusion_pytorch_model.safetensors:  74%|███████████████████████        | 2.56G/3.44G [
diffusion_pytorch_model.safetensors:  75%|███████████████████████▏       | 2.57G/3.44G
diffusion_pytorch_model.safetensors:  75%|███████████████████████▎       | 2.58G/3.44G
diffusion_pytorch_model.safetensors:  75%|███████████████████████▎       | 2.59G/3.44G
diffusion_pytorch_model.safetensors:  76%|███████████████████████▍       | 2.60G/3.44G
diffusion_pytorch_model.safetensors:  76%|███████████████████████▌       | 2.61G/3.44G
diffusion_pytorch_model.safetensors:  76%|███████████████████████▋       | 2.62G/3.44G
diffusion_pytorch_model.safetensors:  77%|███████████████████████▋       | 2.63G/3.44G
diffusion_pytorch_model.safetensors:  77%|███████████████████████▊       | 2.64G/3.44G
diffusion_pytorch_model.safetensors:  77%|███████████████████████▉       | 2.65G/3.44G
diffusion_pytorch_model.safetensors:  77%|████████████████████████       | 2.66G/3.44G
diffusion_pytorch_model.safetensors:  78%|████████████████████████       | 2.67G/3.44G
diffusion_pytorch_model.safetensors:  78%|████████████████████████▏      | 2.68G/3.44G
diffusion_pytorch_model.safetensors:  78%|████████████████████████▎      | 2.69G/3.44G
diffusion_pytorch_model.safetensors:  79%|████████████████████████▍      | 2.71G/3.44G
diffusion_pytorch_model.safetensors:  79%|████████████████████████▍      | 2.72G/3.44G
diffusion_pytorch_model.safetensors:  79%|████████████████████████▌      | 2.73G/3.44G
diffusion_pytorch_model.safetensors:  80%|████████████████████████▋      | 2.74G/3.44G
diffusion_pytorch_model.safetensors:  80%|████████████████████████▊      | 2.75G/3.44G
diffusion_pytorch_model.safetensors:  80%|████████████████████████▊      | 2.76G/3.44G
diffusion_pytorch_model.safetensors:  81%|████████████████████████▉      | 2.77G/3.44G
diffusion_pytorch_model.safetensors:  81%|█████████████████████████      | 2.78G/3.44G
diffusion_pytorch_model.safetensors:  81%|█████████████████████████▏     | 2.79G/3.44
diffusion_pytorch_model.safetensors:  81%|█████████████████████████▏     | 2.80G/3.44
diffusion_pytorch_model.safetensors:  82%|█████████████████████████▎     | 2.81G/3.44
diffusion_pytorch_model.safetensors:  82%|█████████████████████████▍     | 2.82G/3.44
diffusion_pytorch_model.safetensors:  82%|█████████████████████████▌     | 2.83G/3.44
diffusion_pytorch_model.safetensors:  83%|█████████████████████████▌     | 2.84G/3.44
diffusion_pytorch_model.safetensors:  83%|█████████████████████████▋     | 2.85G/3.44
diffusion_pytorch_model.safetensors:  83%|█████████████████████████▊     | 2.86G/3.44
diffusion_pytorch_model.safetensors:  84%|█████████████████████████▉     | 2.87G/3.44
diffusion_pytorch_model.safetensors:  84%|█████████████████████████▉     | 2.88G/3.44
diffusion_pytorch_model.safetensors:  84%|██████████████████████████     | 2.89G/3.44
diffusion_pytorch_model.safetensors:  84%|██████████████████████████▏    | 2.90G/3.4
diffusion_pytorch_model.safetensors:  85%|██████████████████████████▎    | 2.92G/3.4
diffusion_pytorch_model.safetensors:  85%|██████████████████████████▍    | 2.93G/3.4
diffusion_pytorch_model.safetensors:  85%|██████████████████████████▍    | 2.94G/3.4
diffusion_pytorch_model.safetensors:  86%|██████████████████████████▌    | 2.95G/3.4
diffusion_pytorch_model.safetensors:  86%|██████████████████████████▋    | 2.96G/3.4
diffusion_pytorch_model.safetensors:  86%|██████████████████████████▊    | 2.97G/3.4
diffusion_pytorch_model.safetensors:  87%|██████████████████████████▊    | 2.98G/3.4
diffusion_pytorch_model.safetensors:  87%|██████████████████████████▉    | 2.99G/3.4
diffusion_pytorch_model.safetensors:  87%|███████████████████████████    | 3.00G/3.4
diffusion_pytorch_model.safetensors:  88%|███████████████████████████▏   | 3.01G/3.
diffusion_pytorch_model.safetensors:  88%|███████████████████████████▏   | 3.02G/3.
diffusion_pytorch_model.safetensors:  88%|███████████████████████████▎   | 3.03G/3.
diffusion_pytorch_model.safetensors:  88%|███████████████████████████▍   | 3.04G/3.
diffusion_pytorch_model.safetensors:  89%|███████████████████████████▌   | 3.05G/3.
diffusion_pytorch_model.safetensors:  89%|███████████████████████████▌   | 3.06G/3.
diffusion_pytorch_model.safetensors:  89%|███████████████████████████▋   | 3.07G/3.
diffusion_pytorch_model.safetensors:  90%|███████████████████████████▊   | 3.08G/3.
diffusion_pytorch_model.safetensors:  90%|███████████████████████████▉   | 3.09G/3.
diffusion_pytorch_model.safetensors:  90%|███████████████████████████▉   | 3.10G/3.
diffusion_pytorch_model.safetensors:  91%|████████████████████████████   | 3.11G/3.
diffusion_pytorch_model.safetensors:  91%|████████████████████████████▏  | 3.12G/3
diffusion_pytorch_model.safetensors:  91%|████████████████████████████▎  | 3.14G/3
diffusion_pytorch_model.safetensors:  91%|████████████████████████████▎  | 3.15G/3
diffusion_pytorch_model.safetensors:  92%|████████████████████████████▍  | 3.16G/3
diffusion_pytorch_model.safetensors:  92%|████████████████████████████▌  | 3.17G/3
diffusion_pytorch_model.safetensors:  92%|████████████████████████████▋  | 3.18G/3
diffusion_pytorch_model.safetensors:  93%|████████████████████████████▋  | 3.19G/3
diffusion_pytorch_model.safetensors:  93%|████████████████████████████▊  | 3.20G/3
diffusion_pytorch_model.safetensors:  93%|████████████████████████████▉  | 3.21G/3
diffusion_pytorch_model.safetensors:  94%|█████████████████████████████  | 3.22G/3
diffusion_pytorch_model.safetensors:  94%|█████████████████████████████  | 3.23G/3
diffusion_pytorch_model.safetensors:  94%|█████████████████████████████▏ | 3.24G/
diffusion_pytorch_model.safetensors:  95%|█████████████████████████████▎ | 3.25G/
diffusion_pytorch_model.safetensors:  95%|█████████████████████████████▍ | 3.26G/
diffusion_pytorch_model.safetensors:  95%|█████████████████████████████▍ | 3.27G/
diffusion_pytorch_model.safetensors:  95%|█████████████████████████████▌ | 3.28G/
diffusion_pytorch_model.safetensors:  96%|█████████████████████████████▋ | 3.29G/
diffusion_pytorch_model.safetensors:  96%|█████████████████████████████▊ | 3.30G/
diffusion_pytorch_model.safetensors:  96%|█████████████████████████████▉ | 3.31G/
diffusion_pytorch_model.safetensors:  97%|█████████████████████████████▉ | 3.32G/
diffusion_pytorch_model.safetensors:  97%|██████████████████████████████ | 3.33G/
diffusion_pytorch_model.safetensors:  97%|██████████████████████████████▏| 3.34G
diffusion_pytorch_model.safetensors:  98%|██████████████████████████████▎| 3.36G
diffusion_pytorch_model.safetensors:  98%|██████████████████████████████▎| 3.37G
diffusion_pytorch_model.safetensors:  98%|██████████████████████████████▍| 3.38G
diffusion_pytorch_model.safetensors:  99%|██████████████████████████████▌| 3.39G
diffusion_pytorch_model.safetensors: 100%|███████████████████████████████| 3.44G
/3.44G [37:15<00:00, 1.54MB/s]nsors:  99%|██████████████████████████████▋| 3.41G
Fetching 15 files:  87%|███████████████████████████████████████████G
Fetching 15 files: 100%|███████████████████████████████████████████G
██████████| 15/15 [37:18<00:00, 149.21s/it] █████████████████████████| 3.44G
Loading pipeline components...:  14%|██████▏                                    | 1/7 [00:04<00:25,  4.2
Loading pipeline components...:  29%|████████████▎                              | 2/7 [00:05<00:12
Loading pipeline components...:  43%|██████████████████▍                        | 3/7 [00:07
Loading pipeline components...:  71%|██████████████████████████████▋
Loading pipeline components...:  86%|████████████████████████████████████▊
Loading pipeline components...: 100%|█████████████████████████████████████
Loading pipeline components...: 100%|█████████████████████████████████████
██████| 7/7 [00:10<00:00,  1.44s/it]
StableDiffusionPipeline {
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.26.3",
  "_name_or_path": "runwayml/stable-diffusion-v1-5",
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "image_encoder": [
    null,
    null
  ],
  "requires_safety_checker": true,
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}

  2%|█▍                                                                     | 1/50 [09:31<7:46:39, 571.41s/it
  4%|██▊                                                                    | 2/50 [12:33<4:33:47, 342.24s/i
  6%|████▎                                                                  | 3/50 [14:34<3:09:08, 241.46s
  8%|█████▋                                                                 | 4/50 [16:25<2:25:34, 189.89
 10%|███████                                                                | 5/50 [18:42<2:08:00, 170.6
 12%|████████▌                                                              | 6/50 [21:56<2:11:05, 178
 14%|█████████▉                                                             | 7/50 [27:36<2:45:44, 23
 16%|███████████▎                                                           | 8/50 [31:32<2:43:07,
 18%|████████████▊                                                          | 9/50 [34:15<2:24:15,
 20%|██████████████                                                        | 10/50 [37:18<2:14:50
 22%|███████████████▍                                                      | 11/50 [41:00<2:15:
 24%|████████████████▊                                                     | 12/50 [44:47<2:15
 26%|██████████████████▏                                                   | 13/50 [48:35<2:
 28%|███████████████████▌                                                  | 14/50 [52:14<2
 30%|█████████████████████                                                 | 15/50 [56:18<
 32%|█████████████████████▊                                              | 16/50 [1:00:28
 34%|███████████████████████                                             | 17/50 [1:03:4
 36%|████████████████████████▍                                           | 18/50 [1:07
 38%|█████████████████████████▊                                          | 19/50 [1:1
 40%|███████████████████████████▏                                        | 20/50 [1
 42%|████████████████████████████▌                                       | 21/50 [
 44%|█████████████████████████████▉                                      | 22/50
 46%|███████████████████████████████▎                                    | 23/5
 48%|████████████████████████████████▋                                   | 24/
 50%|██████████████████████████████████                                  | 25
 52%|███████████████████████████████████▎                                |
 54%|████████████████████████████████████▋                               |
 56%|███████████████████████████████████████▏

 ```