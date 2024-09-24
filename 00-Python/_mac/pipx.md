

我的新 mac python 環境安裝後，用 pip3 install 會出現下列訊息sudo pipx ensurepath --global

```
cccimac@cccimacdeiMac 04-矩陣 % pip3 install sympy
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try brew install
    xyz, where xyz is the package you are trying to
    install.
    
    If you wish to install a Python library that isn't in Homebrew,
    use a virtual environment:
    
    python3 -m venv path/to/venv
    source path/to/venv/bin/activate
    python3 -m pip install xyz
    
    If you wish to install a Python application that isn't in Homebrew,
    it may be easiest to use 'pipx install xyz', which will manage a
    virtual environment for you. You can install pipx with
    
    brew install pipx
    
    You may restore the old behavior of pip by passing
    the '--break-system-packages' flag to pip, or by adding
    'break-system-packages = true' to your pip.conf file. The latter
    will permanently disable this error.
    
    If you disable this error, we STRONGLY recommend that you additionally
    pass the '--user' flag to pip, or set 'user = true' in your pip.conf
    file. Failure to do this can result in a broken Homebrew installation.
    
    Read more about this behavior here: <https://peps.python.org/pep-0668/>

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
```

根據此訊息，我必須用虛擬環境 venv 跑

但有點麻煩

於是我改用 pipx 來跑

此時你安裝得用 

> pipx install ?package

然後執行得用

> pipx run ?filename

想查看安裝了哪些套件可以用 pipx list

以下是我的一個使用範例

```
cccimac@cccimacdeiMac 02-matplotlib % pipx list
venvs are in /Users/cccimac/.local/pipx/venvs
apps are exposed on your $PATH at /Users/cccimac/.local/bin
manual pages are exposed at /Users/cccimac/.local/share/man
   package matplotlib 3.9.2, installed using Python 3.12.5
    - fonttools
    - pyftmerge
    - pyftsubset
    - ttx
    - man1/ttx.1
   package numpy 2.1.1, installed using Python 3.12.5
    - f2py
    - numpy-config
   package scipy 1.14.1, installed using Python 3.12.5
```


但奇怪的是，執行 matplotlib 有問題 （numpy, scipy, .... 沒問題) 
