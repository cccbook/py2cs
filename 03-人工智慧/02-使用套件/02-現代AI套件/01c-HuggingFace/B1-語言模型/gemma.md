
```
ccckmit@asus MINGW64 /d/ccc/ccc112b/py2cs/03-人工智慧/A3-HuggingFace/01-語言模型 (master)
$ python lm.py
Traceback (most recent call last):
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\utils\_errors.py", line 286, in hf_raise_for_status
    response.raise_for_status()
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\requests\models.py", line 1021, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/google/gemma-7b-it/resolve/main/tokenizer_config.json

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\utils\hub.py", line 385, in cached_file
    resolved_file = hf_hub_download(
                    ^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\utils\_validators.py", line 118, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py", line 1368, in hf_hub_download
    raise head_call_error
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py", line 1238, in hf_hub_download
    metadata = get_hf_file_metadata(
               ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\utils\_validators.py", line 118, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py", line 1631, in get_hf_file_metadata
    r = _request_wrapper(
        ^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py", line 385, in _request_wrapper
    response = _request_wrapper(
               ^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py", line 409, in _request_wrapper
    hf_raise_for_status(response)
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\utils\_errors.py", line 302, in hf_raise_for_status
    raise GatedRepoError(message, response) from e
huggingface_hub.utils._errors.GatedRepoError: 401 Client Error. (Request ID: Root=1-65d71692-711ca355659a916d03640108;2bcec87f-fccc-44b7-8403-b3ca17756f5a)

Cannot access gated repo for url https://huggingface.co/google/gemma-7b-it/resolve/main/tokenizer_config.json.
Repo model google/gemma-7b-it is gated. You must be authenticated to access it.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\ccc\ccc112b\py2cs\03-人工智慧\A3-HuggingFace\01-語言模型\lm.py", line 4, in <module>
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\models\auto\tokenization_auto.py", line 758, in from_pretrained
    tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\models\auto\tokenization_auto.py", line 590, in get_tokenizer_config
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\utils\hub.py", line 400, in cached_file
    raise EnvironmentError(
OSError: You are trying to access a gated repo.
Make sure to request access at https://huggingface.co/google/gemma-7b-it and pass a token having permission to
this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`.

ccckmit@asus MINGW64 /d/ccc/ccc112b/py2cs/03-人工智慧/A3-HuggingFace/01-語言模型 (master)
$ python lm.py
Traceback (most recent call last):
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\utils\_errors.py", line 286, in hf_raise_for_status
    response.raise_for_status()
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\requests\models.py", line 1021, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://huggingface.co/google/gemma-7b-it/resolve/main/tokenizer_config.json

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\utils\hub.py", line 385, in cached_file
    resolved_file = hf_hub_download(
                    ^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\utils\_validators.py", line 118, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py", line 1368, in hf_hub_download
    raise head_call_error
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py", line 1238, in hf_hub_download
    metadata = get_hf_file_metadata(
               ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\utils\_validators.py", line 118, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py", line 1631, in get_hf_file_metadata
    r = _request_wrapper(
        ^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py", line 385, in _request_wrapper
    response = _request_wrapper(
               ^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\file_download.py", line 409, in _request_wrapper
    hf_raise_for_status(response)
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\huggingface_hub\utils\_errors.py", line 302, in hf_raise_for_status
    raise GatedRepoError(message, response) from e
huggingface_hub.utils._errors.GatedRepoError: 403 Client Error. (Request ID: Root=1-65d716c4-7a43f0ca282394a356e80918;c09d8720-d636-4df9-9b63-c2915c60d850)

Cannot access gated repo for url https://huggingface.co/google/gemma-7b-it/resolve/main/tokenizer_config.json.
Access to model google/gemma-7b-it is restricted and you are not in the authorized list. Visit https://huggingface.co/google/gemma-7b-it to ask for access.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\ccc\ccc112b\py2cs\03-人工智慧\A3-HuggingFace\01-語言模型\lm.py", line 4, in <module>
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", token=access_token)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\models\auto\tokenization_auto.py", line 758, in from_pretrained
    tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\models\auto\tokenization_auto.py", line 590, in get_tokenizer_config
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\transformers\utils\hub.py", line 400, in cached_file
    raise EnvironmentError(
OSError: You are trying to access a gated repo.
Make sure to request access at https://huggingface.co/google/gemma-7b-it and pass a token having permission to
this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`.
```
