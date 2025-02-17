# 為了執行 d2l 這本書的範例，請安裝 miniconda

* https://zh.d2l.ai/chapter_installation/index.html


```
conda create --name d2l python=3.9 -y
```

接著開啟 anaconda powershell prompt

創建 d2l 虛擬環境並啟動之

```
conda create --name d2l python=3.9 -y
conda activate d2l
```

然後安裝

```
pip install torch==1.12.0
pip install torchvision==0.13.0
pip install d2l==0.17.6
```

## 執行過程

```
done
#
# To activate this environment, use
#
#     $ conda activate d2l
#
# To deactivate an active environment, use
#
#     $ conda deactivate

(base) PS C:\Users\ccc> conda activate d2l
(d2l) PS C:\Users\ccc> pip install torch==1.12.0
Collecting torch==1.12.0
  Downloading torch-1.12.0-cp39-cp39-win_amd64.whl (161.8 MB)
     ---------------- 161.8/161.8 MB 798.1 kB/s eta 0:00:00
Collecting typing-extensions
  Using cached typing_extensions-4.4.0-py3-none-any.whl (26 kB)
Installing collected packages: typing-extensions, torch
Successfully installed torch-1.12.0 typing-extensions-4.4.0
(d2l) PS C:\Users\ccc> pip install torchvision==0.13.0
Collecting torchvision==0.13.0
  Downloading torchvision-0.13.0-cp39-cp39-win_amd64.whl (1.1 MB)
     ---------------------- 1.1/1.1 MB 1.2 MB/s eta 0:00:00
Requirement already satisfied: torch==1.12.0 in c:\users\ccc\miniconda3\envs\d2l\lib\site-packages (from torchvision==0.13.0) (1.12.0)
Collecting numpy
  Downloading numpy-1.23.5-cp39-cp39-win_amd64.whl (14.7 MB)
     -------------------- 14.7/14.7 MB 1.1 MB/s eta 0:00:00
Requirement already satisfied: typing-extensions in c:\users\ccc\miniconda3\envs\d2l\lib\site-packages (from torchvision==0.13.0) (4.4.0)
Collecting requests
  Using cached requests-2.28.1-py3-none-any.whl (62 kB)
Collecting pillow!=8.3.*,>=5.3.0
  Downloading Pillow-9.3.0-cp39-cp39-win_amd64.whl (2.5 MB)
     -------------------- 2.5/2.5 MB 724.9 kB/s eta 0:00:00
Collecting urllib3<1.27,>=1.21.1
  Downloading urllib3-1.26.13-py2.py3-none-any.whl (140 kB)
     ---------------- 140.6/140.6 kB 396.1 kB/s eta 0:00:00
Collecting idna<4,>=2.5
  Using cached idna-3.4-py3-none-any.whl (61 kB)
Collecting charset-normalizer<3,>=2
  Using cached charset_normalizer-2.1.1-py3-none-any.whl (39 kB)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\ccc\miniconda3\envs\d2l\lib\site-packages (from requests->torchvision==0.13.0) (2022.9.24)
Installing collected packages: urllib3, pillow, numpy, idna, charset-normalizer, requests, torchvision
Successfully installed charset-normalizer-2.1.1 idna-3.4 numpy-1.23.5 pillow-9.3.0 requests-2.28.1 torchvision-0.13.0 urllib3-1.26.13
(d2l) PS C:\Users\ccc> pip install d2l==0.17.6
Collecting d2l==0.17.6
  Downloading d2l-0.17.6-py3-none-any.whl (112 kB)
     ---------------- 112.6/112.6 kB 656.4 kB/s eta 0:00:00
Collecting jupyter==1.0.0
  Downloading jupyter-1.0.0-py2.py3-none-any.whl (2.7 kB)
Collecting matplotlib==3.5.1
  Downloading matplotlib-3.5.1-cp39-cp39-win_amd64.whl (7.2 MB)
     ---------------------- 7.2/7.2 MB 1.3 MB/s eta 0:00:00
Collecting pandas==1.2.4
  Downloading pandas-1.2.4-cp39-cp39-win_amd64.whl (9.3 MB)
     ---------------------- 9.3/9.3 MB 1.4 MB/s eta 0:00:00
Collecting numpy==1.21.5
  Downloading numpy-1.21.5-cp39-cp39-win_amd64.whl (14.0 MB)
     -------------------- 14.0/14.0 MB 1.3 MB/s eta 0:00:00
Collecting requests==2.25.1
  Downloading requests-2.25.1-py2.py3-none-any.whl (61 kB)
     ------------------ 61.2/61.2 kB 463.4 kB/s eta 0:00:00
Collecting ipykernel
  Downloading ipykernel-6.19.2-py3-none-any.whl (145 kB)
     ---------------- 145.1/145.1 kB 539.3 kB/s eta 0:00:00
Collecting qtconsole
  Downloading qtconsole-5.4.0-py3-none-any.whl (121 kB)
     ---------------- 121.0/121.0 kB 589.2 kB/s eta 0:00:00
Collecting notebook
  Downloading notebook-6.5.2-py3-none-any.whl (439 kB)
     ---------------- 439.1/439.1 kB 761.8 kB/s eta 0:00:00
Collecting nbconvert
  Downloading nbconvert-7.2.6-py3-none-any.whl (273 kB)
     ------------------ 273.2/273.2 kB 1.1 MB/s eta 0:00:00
Collecting ipywidgets
  Downloading ipywidgets-8.0.3-py3-none-any.whl (137 kB)
     ---------------- 137.9/137.9 kB 628.7 kB/s eta 0:00:00
Collecting jupyter-console
  Downloading jupyter_console-6.4.4-py3-none-any.whl (22 kB)
Collecting fonttools>=4.22.0
  Using cached fonttools-4.38.0-py3-none-any.whl (965 kB)
Collecting kiwisolver>=1.0.1
  Downloading kiwisolver-1.4.4-cp39-cp39-win_amd64.whl (55 kB)
     ------------------ 55.4/55.4 kB 414.4 kB/s eta 0:00:00
Collecting cycler>=0.10
  Using cached cycler-0.11.0-py3-none-any.whl (6.4 kB)
Collecting pyparsing>=2.2.1
  Using cached pyparsing-3.0.9-py3-none-any.whl (98 kB)
Collecting python-dateutil>=2.7
  Using cached python_dateutil-2.8.2-py2.py3-none-any.whl (247 kB)
Requirement already satisfied: pillow>=6.2.0 in c:\users\ccc\miniconda3\envs\d2l\lib\site-packages (from matplotlib==3.5.1->d2l==0.17.6) (9.3.0)
Collecting packaging>=20.0
  Downloading packaging-22.0-py3-none-any.whl (42 kB)
     ------------------ 42.6/42.6 kB 415.1 kB/s eta 0:00:00
Collecting pytz>=2017.3
  Using cached pytz-2022.6-py2.py3-none-any.whl (498 kB)
Collecting chardet<5,>=3.0.2
  Downloading chardet-4.0.0-py2.py3-none-any.whl (178 kB)
     ------------------ 178.7/178.7 kB 2.2 MB/s eta 0:00:00
Collecting idna<3,>=2.5
  Downloading idna-2.10-py2.py3-none-any.whl (58 kB)
     ------------------ 58.8/58.8 kB 622.7 kB/s eta 0:00:00
Requirement already satisfied: certifi>=2017.4.17 in c:\users\ccc\miniconda3\envs\d2l\lib\site-packages (from requests==2.25.1->d2l==0.17.6) (2022.9.24)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\ccc\miniconda3\envs\d2l\lib\site-packages (from requests==2.25.1->d2l==0.17.6) (1.26.13)
Collecting six>=1.5
  Using cached six-1.16.0-py2.py3-none-any.whl (11 kB)
Collecting traitlets>=5.4.0
  Downloading traitlets-5.7.0-py3-none-any.whl (109 kB)
     ---------------- 109.9/109.9 kB 914.4 kB/s eta 0:00:00
Collecting jupyter-client>=6.1.12
  Downloading jupyter_client-7.4.8-py3-none-any.whl (133 kB)
     ------------------ 133.5/133.5 kB 1.3 MB/s eta 0:00:00
Collecting tornado>=6.1
  Using cached tornado-6.2-cp37-abi3-win_amd64.whl (425 kB)
Collecting nest-asyncio
  Downloading nest_asyncio-1.5.6-py3-none-any.whl (5.2 kB)
Collecting psutil
  Using cached psutil-5.9.4-cp36-abi3-win_amd64.whl (252 kB)
Collecting debugpy>=1.0
  Downloading debugpy-1.6.4-cp39-cp39-win_amd64.whl (4.8 MB)
     ---------------------- 4.8/4.8 MB 1.7 MB/s eta 0:00:00
Collecting pyzmq>=17
  Downloading pyzmq-24.0.1-cp39-cp39-win_amd64.whl (999 kB)
     ------------------ 999.5/999.5 kB 1.9 MB/s eta 0:00:00
Collecting matplotlib-inline>=0.1
  Downloading matplotlib_inline-0.1.6-py3-none-any.whl (9.4 kB)
Collecting ipython>=7.23.1
  Downloading ipython-8.7.0-py3-none-any.whl (761 kB)
     ------------------ 761.7/761.7 kB 1.5 MB/s eta 0:00:00
Collecting comm>=0.1.1
  Downloading comm-0.1.2-py3-none-any.whl (6.5 kB)
Collecting jupyterlab-widgets~=3.0
  Downloading jupyterlab_widgets-3.0.4-py3-none-any.whl (384 kB)
     ------------------ 384.3/384.3 kB 1.4 MB/s eta 0:00:00
Collecting widgetsnbextension~=4.0
  Downloading widgetsnbextension-4.0.4-py3-none-any.whl (2.0 MB)
     ---------------------- 2.0/2.0 MB 1.1 MB/s eta 0:00:00
Collecting pygments
  Using cached Pygments-2.13.0-py3-none-any.whl (1.1 MB)
Collecting prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0
  Downloading prompt_toolkit-3.0.36-py3-none-any.whl (386 kB)
     ---------------- 386.4/386.4 kB 858.8 kB/s eta 0:00:00
Collecting beautifulsoup4
  Downloading beautifulsoup4-4.11.1-py3-none-any.whl (128 kB)
     ------------------ 128.2/128.2 kB 1.3 MB/s eta 0:00:00
Collecting jupyter-core>=4.7
  Downloading jupyter_core-5.1.0-py3-none-any.whl (92 kB)
     ------------------- 92.7/92.7 kB 82.5 kB/s eta 0:00:00
Collecting pandocfilters>=1.4.1
  Downloading pandocfilters-1.5.0-py2.py3-none-any.whl (8.7 kB)
Collecting bleach
  Downloading bleach-5.0.1-py3-none-any.whl (160 kB)
     ---------------- 160.9/160.9 kB 959.6 kB/s eta 0:00:00
Collecting nbclient>=0.5.0
  Downloading nbclient-0.7.2-py3-none-any.whl (71 kB)
     ------------------ 72.0/72.0 kB 493.7 kB/s eta 0:00:00
Collecting mistune<3,>=2.0.3
  Downloading mistune-2.0.4-py2.py3-none-any.whl (24 kB)
Collecting defusedxml
  Downloading defusedxml-0.7.1-py2.py3-none-any.whl (25 kB)
Collecting jupyterlab-pygments
  Downloading jupyterlab_pygments-0.2.2-py2.py3-none-any.whl (21 kB)
Collecting nbformat>=5.1
  Downloading nbformat-5.7.0-py3-none-any.whl (77 kB)
     ------------------ 77.1/77.1 kB 613.4 kB/s eta 0:00:00
Collecting tinycss2
  Downloading tinycss2-1.2.1-py3-none-any.whl (21 kB)
Collecting markupsafe>=2.0
  Downloading MarkupSafe-2.1.1-cp39-cp39-win_amd64.whl (17 kB)
Collecting jinja2>=3.0
  Using cached Jinja2-3.1.2-py3-none-any.whl (133 kB)
Collecting importlib-metadata>=3.6
  Downloading importlib_metadata-5.1.0-py3-none-any.whl (21 kB)
Collecting argon2-cffi
  Downloading argon2_cffi-21.3.0-py3-none-any.whl (14 kB)
Collecting prometheus-client
  Downloading prometheus_client-0.15.0-py3-none-any.whl (60 kB)
     ------------------ 60.1/60.1 kB 804.1 kB/s eta 0:00:00
Collecting terminado>=0.8.3
  Downloading terminado-0.17.1-py3-none-any.whl (17 kB)
Collecting ipython-genutils
  Downloading ipython_genutils-0.2.0-py2.py3-none-any.whl (26 kB)
Collecting Send2Trash>=1.8.0
  Downloading Send2Trash-1.8.0-py3-none-any.whl (18 kB)
Collecting nbclassic>=0.4.7
  Downloading nbclassic-0.4.8-py3-none-any.whl (9.8 MB)
     ---------------------- 9.8/9.8 MB 1.2 MB/s eta 0:00:00
Collecting qtpy>=2.0.1
  Downloading QtPy-2.3.0-py3-none-any.whl (83 kB)
     ------------------ 83.6/83.6 kB 941.0 kB/s eta 0:00:00
Collecting zipp>=0.5
  Downloading zipp-3.11.0-py3-none-any.whl (6.6 kB)
Collecting stack-data
  Downloading stack_data-0.6.2-py3-none-any.whl (24 kB)
Collecting colorama
  Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Collecting decorator
  Using cached decorator-5.1.1-py3-none-any.whl (9.1 kB)
Collecting jedi>=0.16
  Downloading jedi-0.18.2-py2.py3-none-any.whl (1.6 MB)
     -------------------- 1.6/1.6 MB 922.9 kB/s eta 0:00:00
Collecting pickleshare
  Downloading pickleshare-0.7.5-py2.py3-none-any.whl (6.9 kB)
Collecting backcall
  Downloading backcall-0.2.0-py2.py3-none-any.whl (11 kB)
Collecting entrypoints
  Using cached entrypoints-0.4-py3-none-any.whl (5.3 kB)
Collecting pywin32>=1.0
  Downloading pywin32-305-cp39-cp39-win_amd64.whl (12.2 MB)
     ------------------ 12.2/12.2 MB 896.7 kB/s eta 0:00:00
Collecting platformdirs>=2.5
  Downloading platformdirs-2.6.0-py3-none-any.whl (14 kB)
Collecting jupyter-server>=1.8
  Downloading jupyter_server-2.0.1-py3-none-any.whl (360 kB)
     ---------------- 360.5/360.5 kB 423.0 kB/s eta 0:00:00
Collecting notebook-shim>=0.1.0
  Downloading notebook_shim-0.2.2-py3-none-any.whl (13 kB)
Collecting fastjsonschema
  Downloading fastjsonschema-2.16.2-py3-none-any.whl (22 kB)
Collecting jsonschema>=2.6
  Downloading jsonschema-4.17.3-py3-none-any.whl (90 kB)
     ------------------ 90.4/90.4 kB 510.4 kB/s eta 0:00:00
Collecting wcwidth
  Downloading wcwidth-0.2.5-py2.py3-none-any.whl (30 kB)
Collecting pywinpty>=1.1.0
  Downloading pywinpty-2.0.9-cp39-none-win_amd64.whl (1.4 MB)
     -------------------- 1.4/1.4 MB 727.4 kB/s eta 0:00:00
Collecting argon2-cffi-bindings
  Downloading argon2_cffi_bindings-21.2.0-cp36-abi3-win_amd64.whl (30 kB)
Collecting soupsieve>1.2
  Downloading soupsieve-2.3.2.post1-py3-none-any.whl (37 kB)
Collecting webencodings
  Downloading webencodings-0.5.1-py2.py3-none-any.whl (11 kB)
Collecting parso<0.9.0,>=0.8.0
  Downloading parso-0.8.3-py2.py3-none-any.whl (100 kB)
     ---------------- 100.8/100.8 kB 646.7 kB/s eta 0:00:00
Collecting attrs>=17.4.0
  Using cached attrs-22.1.0-py2.py3-none-any.whl (58 kB)
Collecting pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0
  Downloading pyrsistent-0.19.2-cp39-cp39-win_amd64.whl (62 kB)
     ------------------ 62.8/62.8 kB 305.5 kB/s eta 0:00:00
Collecting jupyter-server-terminals
  Downloading jupyter_server_terminals-0.4.2-py3-none-any.whl (13 kB)
Collecting anyio<4,>=3.1.0
  Downloading anyio-3.6.2-py3-none-any.whl (80 kB)
     ------------------ 80.6/80.6 kB 300.8 kB/s eta 0:00:00
Collecting websocket-client
  Using cached websocket_client-1.4.2-py3-none-any.whl (55 kB)
Collecting jupyter-events>=0.4.0
  Downloading jupyter_events-0.5.0-py3-none-any.whl (17 kB)
Collecting cffi>=1.0.1
  Downloading cffi-1.15.1-cp39-cp39-win_amd64.whl (179 kB)
     ---------------- 179.1/179.1 kB 771.0 kB/s eta 0:00:00
Collecting asttokens>=2.1.0
  Downloading asttokens-2.2.1-py2.py3-none-any.whl (26 kB)
Collecting pure-eval
  Downloading pure_eval-0.2.2-py3-none-any.whl (11 kB)
Collecting executing>=1.2.0
  Downloading executing-1.2.0-py2.py3-none-any.whl (24 kB)
Collecting sniffio>=1.1
  Downloading sniffio-1.3.0-py3-none-any.whl (10 kB)
Collecting pycparser
  Using cached pycparser-2.21-py2.py3-none-any.whl (118 kB)
Collecting pyyaml
  Downloading PyYAML-6.0-cp39-cp39-win_amd64.whl (151 kB)
     ---------------- 151.6/151.6 kB 291.6 kB/s eta 0:00:00
Collecting python-json-logger
  Downloading python_json_logger-2.0.4-py3-none-any.whl (7.8 kB)
Collecting jsonpointer>1.13
  Downloading jsonpointer-2.3-py2.py3-none-any.whl (7.8 kB)
Collecting fqdn
  Downloading fqdn-1.5.1-py3-none-any.whl (9.1 kB)
Collecting uri-template
  Downloading uri_template-1.2.0-py3-none-any.whl (10 kB)
Collecting webcolors>=1.11
  Downloading webcolors-1.12-py3-none-any.whl (9.9 kB)
Collecting rfc3986-validator>0.1.0
  Downloading rfc3986_validator-0.1.1-py2.py3-none-any.whl (4.2 kB)
Collecting isoduration
  Downloading isoduration-20.11.0-py3-none-any.whl (11 kB)
Collecting rfc3339-validator
  Downloading rfc3339_validator-0.1.4-py2.py3-none-any.whl (3.5 kB)
Collecting arrow>=0.15.0
  Downloading arrow-1.2.3-py3-none-any.whl (66 kB)
     ------------------ 66.4/66.4 kB 256.4 kB/s eta 0:00:00
Installing collected packages: webencodings, wcwidth, Send2Trash, pywin32, pytz, pure-eval, pickleshare, mistune, ipython-genutils, fastjsonschema, executing, backcall, zipp, widgetsnbextension, websocket-client, webcolors, uri-template, traitlets, tornado, tinycss2, soupsieve, sniffio, six, rfc3986-validator, pyzmq, pyyaml, pywinpty, python-json-logger, pyrsistent, pyparsing, pygments, pycparser, psutil, prompt-toolkit, prometheus-client, platformdirs, parso, pandocfilters, packaging, numpy, nest-asyncio, markupsafe, kiwisolver, jupyterlab-widgets, jupyterlab-pygments, jsonpointer, idna, fqdn, fonttools, entrypoints, defusedxml, decorator, debugpy, cycler, colorama, chardet, attrs, terminado, rfc3339-validator, requests, qtpy, python-dateutil, matplotlib-inline, jupyter-core, jsonschema, jinja2, jedi, importlib-metadata, comm, cffi, bleach, beautifulsoup4, asttokens, anyio, stack-data, pandas, nbformat, matplotlib, jupyter-server-terminals, jupyter-client, arrow, argon2-cffi-bindings, nbclient, isoduration, ipython, argon2-cffi, nbconvert, ipykernel, qtconsole, jupyter-events, jupyter-console, ipywidgets, jupyter-server, notebook-shim, nbclassic, notebook, jupyter, d2l
  Attempting uninstall: numpy
    Found existing installation: numpy 1.23.5
    Uninstalling numpy-1.23.5:
      Successfully uninstalled numpy-1.23.5
  Attempting uninstall: idna
    Found existing installation: idna 3.4
    Uninstalling idna-3.4:
      Successfully uninstalled idna-3.4
  Attempting uninstall: requests
    Found existing installation: requests 2.28.1
    Uninstalling requests-2.28.1:
      Successfully uninstalled requests-2.28.1
Successfully installed Send2Trash-1.8.0 anyio-3.6.2 argon2-cffi-21.3.0 argon2-cffi-bindings-21.2.0 arrow-1.2.3 asttokens-2.2.1 attrs-22.1.0 backcall-0.2.0 beautifulsoup4-4.11.1 bleach-5.0.1 cffi-1.15.1 chardet-4.0.0 colorama-0.4.6 comm-0.1.2 cycler-0.11.0 d2l-0.17.6 debugpy-1.6.4 decorator-5.1.1 defusedxml-0.7.1 entrypoints-0.4 executing-1.2.0 fastjsonschema-2.16.2 fonttools-4.38.0 fqdn-1.5.1 idna-2.10 importlib-metadata-5.1.0 ipykernel-6.19.2 ipython-8.7.0 ipython-genutils-0.2.0 ipywidgets-8.0.3 isoduration-20.11.0 jedi-0.18.2 jinja2-3.1.2 jsonpointer-2.3 jsonschema-4.17.3 jupyter-1.0.0 jupyter-client-7.4.8 jupyter-console-6.4.4 jupyter-core-5.1.0 jupyter-events-0.5.0 jupyter-server-2.0.1 jupyter-server-terminals-0.4.2 jupyterlab-pygments-0.2.2 jupyterlab-widgets-3.0.4 kiwisolver-1.4.4 markupsafe-2.1.1 matplotlib-3.5.1 matplotlib-inline-0.1.6 mistune-2.0.4 nbclassic-0.4.8 nbclient-0.7.2 nbconvert-7.2.6 nbformat-5.7.0 nest-asyncio-1.5.6 notebook-6.5.2 notebook-shim-0.2.2 numpy-1.21.5 packaging-22.0 pandas-1.2.4 pandocfilters-1.5.0 parso-0.8.3 pickleshare-0.7.5 platformdirs-2.6.0 prometheus-client-0.15.0 prompt-toolkit-3.0.36 psutil-5.9.4 pure-eval-0.2.2 pycparser-2.21 pygments-2.13.0 pyparsing-3.0.9 pyrsistent-0.19.2 python-dateutil-2.8.2 python-json-logger-2.0.4 pytz-2022.6 pywin32-305 pywinpty-2.0.9 pyyaml-6.0 pyzmq-24.0.1 qtconsole-5.4.0 qtpy-2.3.0 requests-2.25.1 rfc3339-validator-0.1.4 rfc3986-validator-0.1.1 six-1.16.0 sniffio-1.3.0 soupsieve-2.3.2.post1 stack-data-0.6.2 terminado-0.17.1 tinycss2-1.2.1 tornado-6.2 traitlets-5.7.0 uri-template-1.2.0 wcwidth-0.2.5 webcolors-1.12 webencodings-0.5.1 websocket-client-1.4.2 widgetsnbextension-4.0.4 zipp-3.11.0
```



