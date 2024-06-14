# BERT 安裝

* https://github.com/huggingface/transformers



## 先安裝 venv

```
c:\>c:\Python35\python -m venv c:\path\to\myenv

Alternatively, if you configured the PATH and PATHEXT variables for your Python installation:

c:\>python -m venv c:\path\to\myenv

   48  pip install transformers
   49  c:\users\hero3c\appdata\local\programs\python\python39\python.exe -m pip install --upgrade pip
   50  python -m pip install --upgrade pip
```

我的過程

```
$ python -m venv /c/venv
$ pip install transformers
$ python -m pip install --upgrade pip
```

## 舊:先安裝 venv

* https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

mac

```
mac020:01-BERT mac020$ python3 -m pip install --user --upgrade pip
Collecting pip
  Downloading pip-20.1.1-py2.py3-none-any.whl (1.5 MB)
     |████████████████████████████████| 1.5 MB 558 kB/s 
Installing collected packages: pip
  WARNING: The scripts pip, pip3 and pip3.7 are installed in '/Users/mac020/Library/Python/3.7/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed pip-20.1.1
mac020:01-BERT mac020$ vim ~/.bash_config
mac020:01-BERT mac020$ vim ~/.bash_profile
mac020:01-BERT mac020$ source ~/.bash_profile
mac020:01-BERT mac020$ echo $PATH
/Users/mac020/Library/Python/3.7/bin;/Users/mac020/Desktop/ccc/llvm/bin:/Users/mac020/.deno/bin:/Users/mac020/Desktop/ccc/llvm/bin:/Users/mac020/.deno/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/Users/mac020/Desktop/ccc/llvm/bin:/usr/local/git/bin:/Users/mac020/.deno/bin
mac020:01-BERT mac020$ python3 -m pip --version
pip 20.1.1 from /Users/mac020/Library/Python/3.7/lib/python/site-packages/pip (python 3.7)
mac020:01-BERT mac020$ python3 -m pip install --user virtualenv
Collecting virtualenv
  Downloading virtualenv-20.0.21-py2.py3-none-any.whl (4.7 MB)
     |████████████████████████████████| 4.7 MB 481 kB/s 
Requirement already satisfied: six<2,>=1.9.0 in /usr/local/lib/python3.7/site-packages (from virtualenv) (1.14.0)
Collecting appdirs<2,>=1.4.3
  Downloading appdirs-1.4.4-py2.py3-none-any.whl (9.6 kB)
Collecting distlib<1,>=0.3.0
  Downloading distlib-0.3.0.zip (571 kB)
     |████████████████████████████████| 571 kB 455 kB/s 
Requirement already satisfied: importlib-metadata<2,>=0.12; python_version < "3.8" in /usr/local/lib/python3.7/site-packages (from virtualenv) (1.5.0)
Requirement already satisfied: filelock<4,>=3.0.0 in /usr/local/lib/python3.7/site-packages (from virtualenv) (3.0.12)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/site-packages (from importlib-metadata<2,>=0.12; python_version < "3.8"->virtualenv) (2.2.0)
Building wheels for collected packages: distlib
  Building wheel for distlib (setup.py) ... done
  Created wheel for distlib: filename=distlib-0.3.0-py3-none-any.whl size=340427 sha256=fea6e9dacbecfd2e5ef4874fa829dd21d0f5aafb735864c6f7d001236b3b34c5
  Stored in directory: /Users/mac020/Library/Caches/pip/wheels/a2/19/da/a15d4e2bedf3062c739b190d5cb5b7b2ecfbccb6b0d93c861b
Successfully built distlib
Installing collected packages: appdirs, distlib, virtualenv
  WARNING: The script virtualenv is installed in '/Users/mac020/Library/Python/3.7/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed appdirs-1.4.4 distlib-0.3.0 virtualenv-20.0.21
mac020:01-BERT mac020$ echo $PATH
/Users/mac020/Library/Python/3.7/bin;/Users/mac020/Desktop/ccc/llvm/bin:/Users/mac020/.deno/bin:/Users/mac020/Desktop/ccc/llvm/bin:/Users/mac020/.deno/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/Users/mac020/Desktop/ccc/llvm/bin:/usr/local/git/bin:/Users/mac020/.deno/bin
mac020:01-BERT mac020$ python3 -m venv env
mac020:01-BERT mac020$ source env/bin/activate
(env) mac020:01-BERT mac020$ which python
/Users/mac020/Desktop/ccc/ai2/python/11-deepLearning/11-chinese/01-BERT/env/bin/python
(env) mac020:01-BERT mac020$ deactivate
mac020:01-BERT mac020$ 
```

## 安裝 transformer (BERT)

```
(env) mac020:11-bert mac020$ pip install transformers
Collecting transformers
  Using cached https://files.pythonhosted.org/packages/48/35/ad2c5b1b8f99feaaf9d7cdadaeef261f098c6e1a6a2935d4d07662a6b780/transformers-2.11.0-py3-none-any.whl
Collecting tokenizers==0.7.0 (from transformers)
  Using cached https://files.pythonhosted.org/packages/98/a2/11e6465beaecbf92a3f203e44447a43110e3e0ee2cfdc9cfe03c7e2c1051/tokenizers-0.7.0-cp37-cp37m-macosx_10_10_x86_64.whl
Collecting packaging (from transformers)
  Using cached https://files.pythonhosted.org/packages/46/19/c5ab91b1b05cfe63cccd5cfc971db9214c6dd6ced54e33c30d5af1d2bc43/packaging-20.4-py2.py3-none-any.whl
Collecting sacremoses (from transformers)
  Using cached https://files.pythonhosted.org/packages/7d/34/09d19aff26edcc8eb2a01bed8e98f13a1537005d31e95233fd48216eed10/sacremoses-0.0.43.tar.gz
Collecting filelock (from transformers)
  Using cached https://files.pythonhosted.org/packages/93/83/71a2ee6158bb9f39a90c0dea1637f81d5eef866e188e1971a1b1ab01a35a/filelock-3.0.12-py3-none-any.whl
Collecting regex!=2019.12.17 (from transformers)
  Using cached https://files.pythonhosted.org/packages/b8/7b/01510a6229c2176425bda54d15fba05a4b3df169b87265b008480261d2f9/regex-2020.6.8.tar.gz
Collecting sentencepiece (from transformers)
  Using cached https://files.pythonhosted.org/packages/84/e3/2d755b55423787f438269a26d8bd9743698921fdcde748c6fb050b1c1b8c/sentencepiece-0.1.91-cp37-cp37m-macosx_10_6_x86_64.whl
Collecting numpy (from transformers)
  Using cached https://files.pythonhosted.org/packages/3e/00/0266fefaafb839760d5b25b884375b2ab0f842ebe138ee6c1ef807af44bb/numpy-1.18.5-cp37-cp37m-macosx_10_9_x86_64.whl
Collecting requests (from transformers)
  Using cached https://files.pythonhosted.org/packages/1a/70/1935c770cb3be6e3a8b78ced23d7e0f3b187f5cbfab4749523ed65d7c9b1/requests-2.23.0-py2.py3-none-any.whl
Collecting tqdm>=4.27 (from transformers)
  Using cached https://files.pythonhosted.org/packages/f3/76/4697ce203a3d42b2ead61127b35e5fcc26bba9a35c03b32a2bd342a4c869/tqdm-4.46.1-py2.py3-none-any.whl
Collecting six (from packaging->transformers)
  Using cached https://files.pythonhosted.org/packages/ee/ff/48bde5c0f013094d729fe4b0316ba2a24774b3ff1c52d924a8a4cb04078a/six-1.15.0-py2.py3-none-any.whl
Collecting pyparsing>=2.0.2 (from packaging->transformers)
  Using cached https://files.pythonhosted.org/packages/8a/bb/488841f56197b13700afd5658fc279a2025a39e22449b7cf29864669b15d/pyparsing-2.4.7-py2.py3-none-any.whl
Collecting click (from sacremoses->transformers)
  Using cached https://files.pythonhosted.org/packages/d2/3d/fa76db83bf75c4f8d338c2fd15c8d33fdd7ad23a9b5e57eb6c5de26b430e/click-7.1.2-py2.py3-none-any.whl
Collecting joblib (from sacremoses->transformers)
  Using cached https://files.pythonhosted.org/packages/b8/a6/d1a816b89aa1e9e96bcb298eb1ee1854f21662ebc6d55ffa3d7b3b50122b/joblib-0.15.1-py3-none-any.whl
Collecting certifi>=2017.4.17 (from requests->transformers)
  Using cached https://files.pythonhosted.org/packages/98/99/def511020aa8f663d4a2cfaa38467539e864799289ff354569e339e375b1/certifi-2020.4.5.2-py2.py3-none-any.whl
Collecting chardet<4,>=3.0.2 (from requests->transformers)
  Using cached https://files.pythonhosted.org/packages/bc/a9/01ffebfb562e4274b6487b4bb1ddec7ca55ec7510b22e4c51f14098443b8/chardet-3.0.4-py2.py3-none-any.whl
Collecting urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 (from requests->transformers)
  Using cached https://files.pythonhosted.org/packages/e1/e5/df302e8017440f111c11cc41a6b432838672f5a70aa29227bf58149dc72f/urllib3-1.25.9-py2.py3-none-any.whl
Collecting idna<3,>=2.5 (from requests->transformers)
  Using cached https://files.pythonhosted.org/packages/89/e3/afebe61c546d18fb1709a61bee788254b40e736cff7271c7de5de2dc4128/idna-2.9-py2.py3-none-any.whl
Installing collected packages: tokenizers, six, pyparsing, packaging, regex, click, joblib, tqdm, sacremoses, filelock, sentencepiece, numpy, certifi, chardet, urllib3, idna, requests, transformers
  Running setup.py install for regex ... done
  Running setup.py install for sacremoses ... done
Successfully installed certifi-2020.4.5.2 chardet-3.0.4 click-7.1.2 filelock-3.0.12 idna-2.9 joblib-0.15.1 numpy-1.18.5 packaging-20.4 pyparsing-2.4.7 regex-2020.6.8 requests-2.23.0 sacremoses-0.0.43 sentencepiece-0.1.91 six-1.15.0 tokenizers-0.7.0 tqdm-4.46.1 transformers-2.11.0 urllib3-1.25.9
WARNING: You are using pip version 19.2.3, however version 20.1.1 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
```

## 下載資料集

```
(env) mac020:11-bert mac020$ pwd
/Users/mac020/Desktop/ccc/ai2/python/11-deepLearning/11-bert
(env) mac020:11-bert mac020$ ls dataset
wikitext-2-raw
```
