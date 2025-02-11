

* https://github.com/Jakevin/Speech-OpenAI
    * https://github.com/ccc-ai0/TextRL
    * [Controllable generation via RL to let Elon Musk speak ill of DOGE](https://voidful.dev/jupyter/2022/12/10/textrl-elon-musk.html)
    * Eric Lam=> Python Taiwan -- https://www.facebook.com/groups/197223143437/user/100050120721088/


## install

```
!pip install pfrl@git+https://github.com/voidful/pfrl.git
!pip install textrl==0.1.6
```

## run: on colab

要先選功能表中的《執行階段/變更執行階段類型》然後選 GPU

否則會出現 No CUDA GPUs are available

* https://colab.research.google.com/drive/1iipEEW3DjGAvPduwA1MpFWYxuIfqeiyP?usp=sharing

按  + 程式碼 新增程式區塊

按 Ctrl-Enter 執行

可以看到結果，但我的結果是 ' the best' ，而非原作者文章中的 ' a hoax'

* [Controllable generation via RL to let Elon Musk speak ill of DOGE](https://voidful.dev/jupyter/2022/12/10/textrl-elon-musk.html)

## run: fail on my computer

```
(base) PS D:\ccc\ai\_code\TextRL> python textRL1.py
Traceback (most recent call last):
  File "D:\ccc\ai\_code\TextRL\textRL1.py", line 2, in <module>
    from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelWithLMHead
  File "C:\Users\ccc\miniconda3\lib\site-packages\transformers\__init__.py", line 30, in <module>
    from . import dependency_versions_check
  File "C:\Users\ccc\miniconda3\lib\site-packages\transformers\dependency_versions_check.py", line 17, in <module>
    from .utils.versions import require_version, require_version_core
  File "C:\Users\ccc\miniconda3\lib\site-packages\transformers\utils\__init__.py", line 59, in <module>
    from .hub import (
  File "C:\Users\ccc\miniconda3\lib\site-packages\transformers\utils\hub.py", line 1087, in <module>
    cache_version = int(f.read())
ValueError: invalid literal for int() with base 10: '\x00'
```
