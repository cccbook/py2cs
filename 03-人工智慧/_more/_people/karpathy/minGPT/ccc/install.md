我是在 d2l 書籍的 miniconda 環境中裝起來的 

(不知為何 , pip 環境的 torch 裝不起來 2022/12/12)

```
(d2l) PS D:\ccc\code\py\karpathy\minGPT> pip install -e .
Obtaining file:///D:/ccc/code/py/karpathy/minGPT
  Preparing metadata (setup.py) ... done
Requirement already satisfied: torch in c:\users\ccc\miniconda3\envs\d2l\lib\site-packages (from minGPT==0.0.1) (1.12.0)
Requirement already satisfied: typing-extensions in c:\users\ccc\miniconda3\envs\d2l\lib\site-packages (from torch->minGPT==0.0.1) (4.4.0)
Installing collected packages: minGPT
  Running setup.py develop for minGPT
Successfully installed minGPT-0.0.1
```

## 

```
conda install pytorch
```