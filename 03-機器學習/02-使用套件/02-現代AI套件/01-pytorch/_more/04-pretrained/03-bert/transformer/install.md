
## 安裝

```
(env) mac020:02-chinese mac020$ which python
/Users/mac020/Desktop/ccc/ai2/python/11-deepLearning/11-bert/env/bin/python
(env) mac020:02-chinese mac020$ pip install --upgrade pip
Collecting pip
  Using cached https://files.pythonhosted.org/packages/43/84/23ed6a1796480a6f1a2d38f2802901d078266bda38388954d01d3f2e821d/pip-20.1.1-py2.py3-none-any.whl
Installing collected packages: pip
  Found existing installation: pip 19.2.3
    Uninstalling pip-19.2.3:
      Successfully uninstalled pip-19.2.3
Successfully installed pip-20.1.1
(env) mac020:02-chinese mac020$ pip install transformers tqdm boto3 requests regex -q
(env) mac020:02-chinese mac020$ python bert1.py
/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/Resources/Python.app/Contents/MacOS/Python: can't open file 'bert1.py': [Errno 2] No such file or directory
(env) mac020:02-chinese mac020$ cd ..
(env) mac020:11-bert mac020$ cd 03-use/
(env) mac020:03-use mac020$ python bert1.py
Traceback (most recent call last):
  File "bert1.py", line 1, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
(env) mac020:03-use mac020$ pip install torch
Collecting torch
  Downloading torch-1.5.0-cp37-none-macosx_10_9_x86_64.whl (80.5 MB)
     |████████████████████████████████| 80.5 MB 92 kB/s 
Processing /Users/mac020/Library/Caches/pip/wheels/56/b0/fe/4410d17b32f1f0c3cf54cdfb2bc04d7b4b8f4ae377e2229ba0/future-0.18.2-py3-none-any.whl
Requirement already satisfied: numpy in /Users/mac020/Desktop/ccc/ai2/python/11-deepLearning/11-bert/env/lib/python3.7/site-packages (from torch) (1.18.5)
Installing collected packages: future, torch
Successfully installed future-0.18.2 torch-1.5.0

(env) mac020:03-use mac020$ pip install IPython
Collecting IPython
  Downloading ipython-7.15.0-py3-none-any.whl (783 kB)
     |████████████████████████████████| 783 kB 354 kB/s 
Collecting pexpect; sys_platform != "win32"
  Using cached pexpect-4.8.0-py2.py3-none-any.whl (59 kB)
Collecting appnope; sys_platform == "darwin"
  Using cached appnope-0.1.0-py2.py3-none-any.whl (4.0 kB)
Collecting pygments
  Downloading Pygments-2.6.1-py3-none-any.whl (914 kB)
     |████████████████████████████████| 914 kB 1.6 MB/s 
Processing /Users/mac020/Library/Caches/pip/wheels/9e/56/4f/da13e448a8a5b8671b2954600d5355cf36e557c7aa5020139b/backcall-0.1.0-py3-none-any.whl
Collecting decorator
  Downloading decorator-4.4.2-py2.py3-none-any.whl (9.2 kB)
Collecting jedi>=0.10
  Downloading jedi-0.17.0-py2.py3-none-any.whl (1.1 MB)
     |████████████████████████████████| 1.1 MB 7.0 MB/s 
Requirement already satisfied: setuptools>=18.5 in /Users/mac020/Desktop/ccc/ai2/python/11-deepLearning/11-bert/env/lib/python3.7/site-packages (from IPython) (41.2.0)
Collecting prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0
  Downloading prompt_toolkit-3.0.5-py3-none-any.whl (351 kB)
     |████████████████████████████████| 351 kB 1.5 MB/s 
Collecting pickleshare
  Using cached pickleshare-0.7.5-py2.py3-none-any.whl (6.9 kB)
Collecting traitlets>=4.2
  Using cached traitlets-4.3.3-py2.py3-none-any.whl (75 kB)
Collecting ptyprocess>=0.5
  Using cached ptyprocess-0.6.0-py2.py3-none-any.whl (39 kB)
Collecting parso>=0.7.0
  Downloading parso-0.7.0-py2.py3-none-any.whl (100 kB)
     |████████████████████████████████| 100 kB 1.4 MB/s 
Collecting wcwidth
  Downloading wcwidth-0.2.4-py2.py3-none-any.whl (30 kB)
Collecting ipython-genutils
  Using cached ipython_genutils-0.2.0-py2.py3-none-any.whl (26 kB)
Requirement already satisfied: six in /Users/mac020/Desktop/ccc/ai2/python/11-deepLearning/11-bert/env/lib/python3.7/site-packages (from traitlets>=4.2->IPython) (1.15.0)
Installing collected packages: ptyprocess, pexpect, appnope, pygments, backcall, decorator, parso, jedi, wcwidth, prompt-toolkit, pickleshare, ipython-genutils, traitlets, IPython
Successfully installed IPython-7.15.0 appnope-0.1.0 backcall-0.1.0 decorator-4.4.2 ipython-genutils-0.2.0 jedi-0.17.0 parso-0.7.0 pexpect-4.8.0 pickleshare-0.7.5 prompt-toolkit-3.0.5 ptyprocess-0.6.0 pygments-2.6.1 traitlets-4.3.3 wcwidth-0.2.4
```
