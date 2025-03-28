
* https://pypi.org/project/pydeepl/

* pip install pydeepl

## 執行結果 -- 失敗

```
mac020:deepL mac020$ python3 pydeepL1.py
Traceback (most recent call last):
  File "pydeepL1.py", line 7, in <module>
    translation = pydeepl.translate(sentence, to_language, from_lang=from_language)
  File "/usr/local/lib/python3.7/site-packages/pydeepl/pydeepl.py", line 97, in translate
    raise TranslationError('DeepL call resulted in a unknown result.')
pydeepl.pydeepl.TranslationError: DeepL call resulted in a unknown result.
```



