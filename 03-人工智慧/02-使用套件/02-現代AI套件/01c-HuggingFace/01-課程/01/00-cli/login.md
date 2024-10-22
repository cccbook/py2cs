
* https://huggingface.co/docs/huggingface_hub/en/guides/cli


    pip install -U "huggingface_hub[cli]"

    huggingface-cli login

使用模型有時需授權

例如

* https://huggingface.co/meta-llama/Llama-3.2-1B

你必須看完授權書後填寫基本資料，然後按同意才能用。

## run

```
% pip install -U "huggingface_hub[cli]"
Requirement already satisfied: huggingface_hub[cli] in /opt/miniconda3/lib/python3.12/site-packages (0.25.1)
Collecting huggingface_hub[cli]
  Downloading huggingface_hub-0.25.2-py3-none-any.whl.metadata (13 kB)
Requirement already satisfied: filelock in /opt/miniconda3/lib/python3.12/site-packages (from huggingface_hub[cli]) (3.16.1)
Requirement already satisfied: fsspec>=2023.5.0 in /opt/miniconda3/lib/python3.12/site-packages (from huggingface_hub[cli]) (2024.6.1)
Requirement already satisfied: packaging>=20.9 in /opt/miniconda3/lib/python3.12/site-packages (from huggingface_hub[cli]) (24.1)
Requirement already satisfied: pyyaml>=5.1 in /opt/miniconda3/lib/python3.12/site-packages (from huggingface_hub[cli]) (6.0.2)
Requirement already satisfied: requests in /opt/miniconda3/lib/python3.12/site-packages (from huggingface_hub[cli]) (2.32.3)
Requirement already satisfied: tqdm>=4.42.1 in /opt/miniconda3/lib/python3.12/site-packages (from huggingface_hub[cli]) (4.66.4)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/miniconda3/lib/python3.12/site-packages (from huggingface_hub[cli]) (4.12.2)
Collecting InquirerPy==0.3.4 (from huggingface_hub[cli])
  Downloading InquirerPy-0.3.4-py3-none-any.whl.metadata (8.1 kB)
Collecting pfzy<0.4.0,>=0.3.1 (from InquirerPy==0.3.4->huggingface_hub[cli])
  Downloading pfzy-0.3.4-py3-none-any.whl.metadata (4.9 kB)
Collecting prompt-toolkit<4.0.0,>=3.0.1 (from InquirerPy==0.3.4->huggingface_hub[cli])
  Downloading prompt_toolkit-3.0.48-py3-none-any.whl.metadata (6.4 kB)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/miniconda3/lib/python3.12/site-packages (from requests->huggingface_hub[cli]) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /opt/miniconda3/lib/python3.12/site-packages (from requests->huggingface_hub[cli]) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/miniconda3/lib/python3.12/site-packages (from requests->huggingface_hub[cli]) (2.2.2)
Requirement already satisfied: certifi>=2017.4.17 in /opt/miniconda3/lib/python3.12/site-packages (from requests->huggingface_hub[cli]) (2024.8.30)
Collecting wcwidth (from prompt-toolkit<4.0.0,>=3.0.1->InquirerPy==0.3.4->huggingface_hub[cli])
  Downloading wcwidth-0.2.13-py2.py3-none-any.whl.metadata (14 kB)
Downloading InquirerPy-0.3.4-py3-none-any.whl (67 kB)
Downloading huggingface_hub-0.25.2-py3-none-any.whl (436 kB)
Downloading pfzy-0.3.4-py3-none-any.whl (8.5 kB)
Downloading prompt_toolkit-3.0.48-py3-none-any.whl (386 kB)
Downloading wcwidth-0.2.13-py2.py3-none-any.whl (34 kB)
Installing collected packages: wcwidth, prompt-toolkit, pfzy, InquirerPy, huggingface_hub
  Attempting uninstall: huggingface_hub
    Found existing installation: huggingface-hub 0.25.1
    Uninstalling huggingface-hub-0.25.1:
      Successfully uninstalled huggingface-hub-0.25.1
Successfully installed InquirerPy-0.3.4 huggingface_hub-0.25.2 pfzy-0.3.4 prompt-toolkit-3.0.48 wcwidth-0.2.13
(base) cccimac@cccimacdeiMac 03-textgen % huggingface-cli login

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
Add token as git credential? (Y/n) Y
Token is valid (permission: fineGrained).
Your token has been saved in your configured git credential helpers (osxkeychain).
Your token has been saved to /Users/cccimac/.cache/huggingface/token
Login successful

(base) cccimac@cccimacdeiMac 03-textgen % python textGen1.py
Traceback (most recent call last):
  File "/opt/miniconda3/lib/python3.12/site-packages/huggingface_hub/utils/_http.py", line 406, in hf_raise_for_status
    response.raise_for_status()
  File "/opt/miniconda3/lib/python3.12/site-packages/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/miniconda3/lib/python3.12/site-packages/transformers/utils/hub.py", line 402, in cached_file
    resolved_file = hf_hub_download(
                    ^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/huggingface_hub/utils/_deprecation.py", line 101, in inner_f
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 1232, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 1339, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "/opt/miniconda3/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 1854, in _raise_on_head_call_error
    raise head_call_error
  File "/opt/miniconda3/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 1746, in _get_metadata_or_catch_error
    metadata = get_hf_file_metadata(
               ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 1666, in get_hf_file_metadata
    r = _request_wrapper(
        ^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 364, in _request_wrapper
    response = _request_wrapper(
               ^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 388, in _request_wrapper
    hf_raise_for_status(response)
  File "/opt/miniconda3/lib/python3.12/site-packages/huggingface_hub/utils/_http.py", line 423, in hf_raise_for_status
    raise _format(GatedRepoError, message, response) from e
huggingface_hub.errors.GatedRepoError: 403 Client Error. (Request ID: Root=1-670c93df-7539ab2121417489427fbeba;49f1cc88-b8a3-4cdc-a711-1312b9099e8e)

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json.
Your request to access model meta-llama/Llama-3.2-1B is awaiting a review from the repo authors.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/cccimac/Desktop/ccc/py2cs/03-人工智慧/07b-語言模型/_HF課程/01/03-textgen/textGen1.py", line 4, in <module>
    generator = pipeline(
                ^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/transformers/pipelines/__init__.py", line 805, in pipeline
    config = AutoConfig.from_pretrained(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/transformers/models/auto/configuration_auto.py", line 976, in from_pretrained
    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/transformers/configuration_utils.py", line 632, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/transformers/configuration_utils.py", line 689, in _get_config_dict
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "/opt/miniconda3/lib/python3.12/site-packages/transformers/utils/hub.py", line 420, in cached_file
    raise EnvironmentError(
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.2-1B.
403 Client Error. (Request ID: Root=1-670c93df-7539ab2121417489427fbeba;49f1cc88-b8a3-4cdc-a711-1312b9099e8e)

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json.
Your request to access model meta-llama/Llama-3.2-1B is awaiting a review from the repo authors.
```