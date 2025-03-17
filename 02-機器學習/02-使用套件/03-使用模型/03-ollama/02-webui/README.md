# docker

## 安裝好 docker


* [ChatGPT高仿版WebUI：Ollama + Open WebUI本地环境搭建](https://www.youtube.com/watch?v=VfeCri0yplc)

* https://docs.openwebui.com/getting-started/

## 啟動 docker

到 mac 的應用程式去按 docker

## 啟動 open-webui

```
$ docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

## 第二次啟動

```
(base) cccimac@cccimacdeiMac law % docker images
REPOSITORY                      TAG       IMAGE ID       CREATED      SIZE
ghcr.io/open-webui/open-webui   main      45e146cb5841   8 days ago   3.14GB
```


```
(base) cccimac@cccimacdeiMac law % docker run 45e1      
Loading WEBUI_SECRET_KEY from file, not provided as an environment variable.
Generating WEBUI_SECRET_KEY
Loading WEBUI_SECRET_KEY from .webui_secret_key
/app/backend/open_webui
/app/backend
/app
Running migrations
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 7e5b5dc7342b, init
INFO  [alembic.runtime.migration] Running upgrade 7e5b5dc7342b -> ca81bd47c050, Add config table
INFO  [open_webui.env] 'DEFAULT_LOCALE' loaded from the latest database entry
INFO  [open_webui.env] 'DEFAULT_PROMPT_SUGGESTIONS' loaded from the latest database entry
WARNI [open_webui.env] 

WARNING: CORS_ALLOW_ORIGIN IS SET TO '*' - NOT RECOMMENDED FOR PRODUCTION DEPLOYMENTS.

INFO  [open_webui.env] Embedding model set: sentence-transformers/all-MiniLM-L6-v2
INFO  [open_webui.apps.audio.main] whisper_device_type: cpu
WARNI [langchain_community.utils.user_agent] USER_AGENT environment variable not set, consider setting it to identify your requests.
/usr/local/lib/python3.11/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO  [open_webui.apps.openai.main] get_all_models()
INFO  [open_webui.apps.ollama.main] get_all_models()
```