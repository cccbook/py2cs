

(minGPT) PS D:\ccc\code\py\karpathy\minGPT\ccc\02-generate> python generate.py
Traceback (most recent call last):
  File "D:\ccc\code\py\karpathy\minGPT\ccc\02-generate\generate.py", line 3, in <module>
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
  File "C:\Users\ccc\miniconda3\envs\minGPT\lib\site-packages\transformers\__init__.py", line 30, in <module>
    from . import dependency_versions_check
  File "C:\Users\ccc\miniconda3\envs\minGPT\lib\site-packages\transformers\dependency_versions_check.py", line 17, in <module>
    from .utils.versions import require_version, require_version_core
  File "C:\Users\ccc\miniconda3\envs\minGPT\lib\site-packages\transformers\utils\__init__.py", line 59, in <module>
    from .hub import (
  File "C:\Users\ccc\miniconda3\envs\minGPT\lib\site-packages\transformers\utils\hub.py", line 1087, in <module>
    cache_version = int(f.read())
ValueError: invalid literal for int() with base 10: '\x00'