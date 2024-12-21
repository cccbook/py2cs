# SAT

## expEval.js

```
$ python expEval.py
exp= (x or not z) assign= (0, 1, 1) result= False
```

## globalEval.js

```
$ python globalEval.py
exp= (x or not z) assign= (0, 1, 1) result= False
g= {'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x000002DA1FB52948>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': 'globalEval.py', '__cached__': None, 'g': {...}, 'assign': (0, 1, 1), 'x': 0, 'y': 1, 'z': 1, 'exp': '(x or not z)', 'result': False}
```

## sat.py

```sh
$ python sat.py
exp= (x or y) and (not x or not z) and (x) and (y)
['x', 'y', 'z']   
[0, 0, 0] => 0    
[0, 0, 1] => 0    
[0, 1, 0] => 0    
[0, 1, 1] => 0    
[1, 0, 0] => 0    
[1, 0, 1] => False
[1, 1, 0] => 1    
{'answer': [1, 1, 0]}
exp= (x) and (not x) and (not y) and (not z)
['x', 'y', 'z']
[0, 0, 0] => 0
[0, 0, 1] => 0
[0, 1, 0] => 0
[0, 1, 1] => 0
[1, 0, 0] => False
[1, 0, 1] => False
[1, 1, 0] => False
[1, 1, 1] => False
```

## 參考

* https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/globalThis
* https://stackoverflow.com/questions/62004574/how-to-execute-deno-without-strict-mode-error-uncaught-syntaxerror-identifie