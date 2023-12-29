# 量子程式設計


## QuTiP


* http://qutip.org/
    * QuTiP is open-source software for simulating the dynamics of open quantum systems. 

* [Lecture on QuTip](http://jrjohansson.github.io/computing.html)

## 安裝

```
$ pip install qutip
```

結果安裝失敗

```
PS D:\ccc\course\ai\python\08-scientific\quantum> pip install qutip 
Collecting qutip
  Using cached https://files.pythonhosted.org/packages/61/ae/96cd2050155b125b70be182f13d052746e9ee000e93f466ab1de4e2b9c1f/qutip-4.4.1.tar.gz
Requirement already satisfied: numpy>=1.12 in c:\users\user\appdata\local\programs\python\python37\lib\site-packages (from qutip) (1.14.6)
Requirement already satisfied: scipy>=1.0 in c:\users\user\appdata\local\programs\python\python37\lib\site-packages (from qutip) (1.3.1)
Requirement already satisfied: cython>=0.21 in c:\users\user\appdata\local\programs\python\python37\lib\site-packages (from qutip) (0.29.14)
Building wheels for collected packages: qutip
  Building wheel for qutip (setup.py) ... error
  ERROR: Command errored out with exit status 1:
   command: 'c:\users\user\appdata\local\programs\python\python37\python.exe' -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'C:\\Users\\user\\AppData\\Local\\Temp\\pip-install-sutcvhrp\\qutip\\setup.py'"'"'; __file__='"'"'C:\\Users\\user\\AppData\\Local\\Temp\\pip-install-sutcvhrp\\qutip\\setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' bdist_wheel -d 'C:\Users\user\AppData\Local\Temp\pip-wheel-d9cuactj' --python-tag cp37
       cwd: C:\Users\user\AppData\Local\Temp\pip-install-sutcvhrp\qutip\
  Complete output (294 lines):
  running bdist_wheel
  running build
  running build_py
  creating build
  creating build\lib.win-amd64-3.7
  creating build\lib.win-amd64-3.7\qutip
  copying qutip\about.py -> build\lib.win-amd64-3.7\qutip
  copying qutip\bloch.py -> build\lib.win-amd64-3.7\qutip
  copying qutip\bloch3d.py -> build\lib.win-amd64-3.7\qutip
```