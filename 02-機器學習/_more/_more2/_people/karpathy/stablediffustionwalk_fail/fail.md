(minGPT) PS D:\ccc\ai\_diy\karpathy\stable_diffustion_walk> python .\stablediffusionwalk.py
Traceback (most recent call last):
  File "D:\ccc\ai\_diy\karpathy\stable_diffustion_walk\stablediffusionwalk.py", line 17, in <module>
    from diffusers import StableDiffusionPipeline
  File "C:\Users\ccc\miniconda3\envs\minGPT\lib\site-packages\diffusers\__init__.py", line 38, in <module>
    from .pipeline_utils import DiffusionPipeline
  File "C:\Users\ccc\miniconda3\envs\minGPT\lib\site-packages\diffusers\pipeline_utils.py", line 55, in <module>
    import transformers
  File "C:\Users\ccc\miniconda3\envs\minGPT\lib\site-packages\transformers\__init__.py", line 30, in <module>
    from . import dependency_versions_check
  File "C:\Users\ccc\miniconda3\envs\minGPT\lib\site-packages\transformers\dependency_versions_check.py", line 17, in <module>
    from .utils.versions import require_version, require_version_core
  File "C:\Users\ccc\miniconda3\envs\minGPT\lib\site-packages\transformers\utils\__init__.py", line 59, in <module>
    from .hub import (
  File "C:\Users\ccc\miniconda3\envs\minGPT\lib\site-packages\transformers\utils\hub.py", line 1087, in <module>
    cache_version = int(f.read())
ValueError: invalid literal for int() with base 10: '\x00'