
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
```