
(base) cccimac@cccimacdeiMac 03-callImac % ./reverseOllama.sh 
root@139.162.90.34's password: 
Welcome to Ubuntu 24.04.1 LTS (GNU/Linux 6.8.0-41-generic x86_64)

root@localhost:~# git clone git@github.com:ccc-py/fastai.git
Cloning into 'fastai'...
remote: Enumerating objects: 53, done.
remote: Counting objects: 100% (53/53), done.
remote: Compressing objects: 100% (37/37), done.
remote: Total 53 (delta 13), reused 47 (delta 10), pack-reused 0 (from 0)
Receiving objects: 100% (53/53), 298.22 KiB | 433.00 KiB/s, done.
Resolving deltas: 100% (13/13), done.
root@localhost:~# ls
fastai
root@localhost:~# cd fastai
root@localhost:~/fastai# ls
doc  LICENSE  README.md  test
root@localhost:~/fastai# cd test
root@localhost:~/fastai/test# ls
01-ollamaApi  02-fastapi1  03-callImac  A1-websocket
root@localhost:~/fastai/test# cd 03-callImac/
root@localhost:~/fastai/test/03-callImac# ls
chatGpt.md  ollamaApi.py  README.md  reverseOllama.sh  test.md
root@localhost:~/fastai/test/03-callImac# ls
chatGpt.md  ollamaApi.py  README.md  reverseOllama.sh  test.md
root@localhost:~/fastai/test/03-callImac# python ollamaApi.py
Command 'python' not found, did you mean:
  command 'python3' from deb python3
  command 'python' from deb python-is-python3
root@localhost:~/fastai/test/03-callImac# python3
Python 3.12.3 (main, Sep 11 2024, 14:17:37) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> quit()
root@localhost:~/fastai/test/03-callImac# ls
chatGpt.md  ollamaApi.py  README.md  reverseOllama.sh  test.md
root@localhost:~/fastai/test/03-callImac# python3 ollamaApi.py
我是一個 AI 語言模型，專門在與您互動。

我是一個程式設計的語言模型，可以理解和回應您的自然語言指令。root@localhost:~/fastai/test/03-callImac# 