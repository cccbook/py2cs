# 優化算法

1. [爬山演算法](hillClimbing.md)
2. [遺傳演算法](geneticAlgorithm.md)

## 安裝 pip

```
PS D:\ccc\ai\python\02-optimize> python hillClimbingEquation.py
Traceback (most recent call last):
  File "hillClimbingEquation.py", line 2, in <module>
    from solutionEquation import SolutionEquation # 引入平方根解答類別   
  File "D:\ccc\ai\python\02-optimize\solutionEquation.py", line 12, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
PS D:\ccc\ai\python\02-optimize> pip install numpy
pip : 無法辨識 'pip' 詞彙是否為 Cmdlet、函數、指令檔或可執行程式的名稱。
次。
位於 線路:1 字元:1
+ pip install numpy
+ ~~~
    + CategoryInfo          : ObjectNotFound: (pip:String) [], CommandN  
   otFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
 
PS D:\ccc\ai\python\02-optimize> python get-pip.py
Collecting pip
  Downloading pip-20.0.2-py2.py3-none-any.whl (1.4 MB)
     |████████████████████████████████| 1.4 MB 1.3 MB/s 
Collecting wheel
  Downloading wheel-0.34.2-py2.py3-none-any.whl (26 kB)
Installing collected packages: pip, wheel
  Attempting uninstall: pip
    Found existing installation: pip 19.2.3
    Uninstalling pip-19.2.3:
      Successfully uninstalled pip-19.2.3
  WARNING: The scripts pip.exe, pip3.8.exe and pip3.exe are installed in 
'C:\Users\Tim\AppData\Local\Programs\Python\Python38\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script wheel.exe is installed in 'C:\Users\Tim\AppData\Local\Programs\Python\Python38\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed pip-20.0.2 wheel-0.34.2
```

## 安裝 numpy

```
PS D:\ccc\ai\python\02-optimize> pip install numpy
Collecting numpy
  Downloading numpy-1.18.1-cp38-cp38-win_amd64.whl (12.8 MB)
     |████████████████████████████████| 12.8 MB 383 kB/s 
Installing collected packages: numpy
Successfully installed numpy-1.18.1
```
