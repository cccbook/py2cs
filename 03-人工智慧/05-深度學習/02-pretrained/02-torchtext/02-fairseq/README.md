# firseq

* https://github.com/pytorch/fairseq

## en2de.py

失敗

```
PS D:\pmedia\陳鍾誠\課程\人工智慧\08-deep\02-pretrained\02-torchtext\02-fairseq> python en2de.py
Downloading: "https://github.com/pytorch/fairseq/archive/master.zip" to C:\Users\user/.cache\torch\hub\master.zip
Traceback (most recent call last):
  File "en2de.py", line 2, in <module>
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
  File "C:\Users\user\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\hub.py", line 339, in load       
    model = _load_local(repo_or_dir, model, *args, **kwargs)
  File "C:\Users\user\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\hub.py", line 365, in _load_local
    hub_module = import_module(MODULE_HUBCONF, hubconf_path)
  File "C:\Users\user\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\hub.py", line 74, in import_module
    spec.loader.exec_module(module)
  File "<frozen importlib._bootstrap_external>", line 783, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "C:\Users\user/.cache\torch\hub\pytorch_fairseq_master\hubconf.py", line 35, in <module>
    raise RuntimeError("Missing dependencies: {}".format(", ".join(missing_deps)))
RuntimeError: Missing dependencies: hydra-core, omegaconf
```