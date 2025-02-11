# autotest

## run 

```
(base) teacher@teacherdeiMac 01-hello % uvicorn main:app --reload
INFO:     Will watch for changes in these directories: ['/Users/teacher/Desktop/ccc/py2cs/09-應用程式/08-fastapi/01-hello']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [2554] using WatchFiles
INFO:     Started server process [2556]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     127.0.0.1:49771 - "GET / HTTP/1.1" 200 OK
```

## test


```
(base) teacher@teacherdeiMac 02-uvicorn % pytest
=================== test session starts ====================
platform darwin -- Python 3.11.7, pytest-7.4.0, pluggy-1.0.0
rootdir: /Users/teacher/Desktop/ccc/py2cs/09-應用程式/08-fastapi/02-uvicorn
plugins: anyio-4.2.0
collected 3 items                                          

test_main.py ...                                     [100%]

==================== 3 passed in 0.21s =====================
```