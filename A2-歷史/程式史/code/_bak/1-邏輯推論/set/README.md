# 羅素集合悖論

```
d:\ccc109\se\python\alg\18-unsolvable\set>python russellSet.py
Traceback (most recent call last):
  File "russellSet.py", line 11, in <module>
    print('A.has(A)=', A.has(A))
  File "russellSet.py", line 7, in has
    return not e.has(e)
  File "russellSet.py", line 7, in has
    return not e.has(e)
  File "russellSet.py", line 7, in has
    return not e.has(e)
  [Previous line repeated 996 more times]
RecursionError: maximum recursion depth exceeded
```