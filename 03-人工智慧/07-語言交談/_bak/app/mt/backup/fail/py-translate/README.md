# py-translate

## 執行--失敗

```
mac020:py-translate mac020$ pip install py-translate
Collecting py-translate
  Downloading py_translate-1.0.3-py2.py3-none-any.whl (61 kB)
     |████████████████████████████████| 61 kB 113 kB/s 
Installing collected packages: py-translate
Successfully installed py-translate-1.0.3
mac020:py-translate mac020$ translate en zh-TW <<< 'Hello World!'
Traceback (most recent call last):
  File "/usr/local/lib/python3.7/site-packages/translate/coroutines.py", line 170, in spool
    stream = yield
GeneratorExit

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.7/site-packages/translate/coroutines.py", line 145, in set_task
    task = yield
GeneratorExit

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/bin/translate", line 8, in <module>
    sys.exit(main())
  File "/usr/local/lib/python3.7/site-packages/translate/__main__.py", line 115, in main
    return source(spool(set_task(translate, translit=args.translit)), args.text)
  File "/usr/local/lib/python3.7/site-packages/translate/coroutines.py", line 204, in source
    return target.close()
  File "/usr/local/lib/python3.7/site-packages/translate/coroutines.py", line 180, in spool
    iterable.close()
  File "/usr/local/lib/python3.7/site-packages/translate/coroutines.py", line 149, in set_task
    list(map(stream, workers.map(translator, queue)))
  File "/usr/local/lib/python3.7/site-packages/translate/coroutines.py", line 103, in write_stream
    sentence, _ = script
ValueError: too many values to unpack (expected 2)

```
