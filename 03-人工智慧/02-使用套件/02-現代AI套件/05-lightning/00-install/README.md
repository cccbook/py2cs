

```
% pip install lightning
Collecting lightning
  Downloading lightning-2.4.0-py3-none-any.whl.metadata (38 kB)
Requirement already satisfied: PyYAML<8.0,>=5.4 in /opt/miniconda3/lib/python3.12/site-packages (from lightning) (6.0.2)
Requirement already satisfied: fsspec<2026.0,>=2022.5.0 in /opt/miniconda3/lib/python3.12/site-packages (from fsspec[http]<2026.0,>=2022.5.0->lightning) (2024.6.1)
Collecting lightning-utilities<2.0,>=0.10.0 (from lightning)
  Downloading lightning_utilities-0.11.8-py3-none-any.whl.metadata (5.2 kB)
Requirement already satisfied: packaging<25.0,>=20.0 in /opt/miniconda3/lib/python3.12/site-packages (from lightning) (24.1)
Requirement already satisfied: torch<4.0,>=2.1.0 in /opt/miniconda3/lib/python3.12/site-packages (from lightning) (2.4.1)
Collecting torchmetrics<3.0,>=0.7.0 (from lightning)
  Downloading torchmetrics-1.4.3-py3-none-any.whl.metadata (19 kB)
Requirement already satisfied: tqdm<6.0,>=4.57.0 in /opt/miniconda3/lib/python3.12/site-packages (from lightning) (4.66.4)
Requirement already satisfied: typing-extensions<6.0,>=4.4.0 in /opt/miniconda3/lib/python3.12/site-packages (from lightning) (4.12.2)
Collecting pytorch-lightning (from lightning)
  Downloading pytorch_lightning-2.4.0-py3-none-any.whl.metadata (21 kB)
Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /opt/miniconda3/lib/python3.12/site-packages (from fsspec[http]<2026.0,>=2022.5.0->lightning) (3.10.6)
Requirement already satisfied: setuptools in /opt/miniconda3/lib/python3.12/site-packages (from lightning-utilities<2.0,>=0.10.0->lightning) (72.1.0)
Requirement already satisfied: filelock in /opt/miniconda3/lib/python3.12/site-packages (from torch<4.0,>=2.1.0->lightning) (3.16.1)
Requirement already satisfied: sympy in /opt/miniconda3/lib/python3.12/site-packages (from torch<4.0,>=2.1.0->lightning) (1.13.3)
Requirement already satisfied: networkx in /opt/miniconda3/lib/python3.12/site-packages (from torch<4.0,>=2.1.0->lightning) (3.3)
Requirement already satisfied: jinja2 in /opt/miniconda3/lib/python3.12/site-packages (from torch<4.0,>=2.1.0->lightning) (3.1.4)
Requirement already satisfied: numpy>1.20.0 in /opt/miniconda3/lib/python3.12/site-packages (from torchmetrics<3.0,>=0.7.0->lightning) (1.26.4)
Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /opt/miniconda3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2026.0,>=2022.5.0->lightning) (2.4.0)
Requirement already satisfied: aiosignal>=1.1.2 in /opt/miniconda3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2026.0,>=2022.5.0->lightning) (1.3.1)
Requirement already satisfied: attrs>=17.3.0 in /opt/miniconda3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2026.0,>=2022.5.0->lightning) (24.2.0)
Requirement already satisfied: frozenlist>=1.1.1 in /opt/miniconda3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2026.0,>=2022.5.0->lightning) (1.4.1)
Requirement already satisfied: multidict<7.0,>=4.5 in /opt/miniconda3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2026.0,>=2022.5.0->lightning) (6.1.0)
Requirement already satisfied: yarl<2.0,>=1.12.0 in /opt/miniconda3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2026.0,>=2022.5.0->lightning) (1.12.1)
Requirement already satisfied: MarkupSafe>=2.0 in /opt/miniconda3/lib/python3.12/site-packages (from jinja2->torch<4.0,>=2.1.0->lightning) (2.1.5)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/miniconda3/lib/python3.12/site-packages (from sympy->torch<4.0,>=2.1.0->lightning) (1.3.0)
Requirement already satisfied: idna>=2.0 in /opt/miniconda3/lib/python3.12/site-packages (from yarl<2.0,>=1.12.0->aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2026.0,>=2022.5.0->lightning) (3.7)
Downloading lightning-2.4.0-py3-none-any.whl (810 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━ 811.0/811.0 kB 1.7 MB/s eta 0:00:00
Downloading lightning_utilities-0.11.8-py3-none-any.whl (26 kB)
Downloading torchmetrics-1.4.3-py3-none-any.whl (869 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━ 869.5/869.5 kB 2.2 MB/s eta 0:00:00
Downloading pytorch_lightning-2.4.0-py3-none-any.whl (815 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━ 815.2/815.2 kB 2.3 MB/s eta 0:00:00
Installing collected packages: lightning-utilities, torchmetrics, pytorch-lightning, lightning
Successfully installed lightning-2.4.0 lightning-utilities-0.11.8 pytorch-lightning-2.4.0 torchmetrics-1.4.3
```