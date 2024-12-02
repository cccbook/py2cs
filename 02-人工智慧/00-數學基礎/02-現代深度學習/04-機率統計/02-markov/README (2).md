# 馬可夫鏈 Markov Chain

## markov.py

計算某序列 [b,a,b,b] 的機率

```sh
$ python markov.py
P(['b', 'a', 'b', 'b'])=0.06
```

## mcmc.py

```sh
user@DESKTOP-96FRN6B MINGW64 /d/ccc109/se/python/alg/03-monteCarlo/markov (master)
$ python mcmc.py
P(['b', 'a', 'b', 'b'])=0.06131

user@DESKTOP-96FRN6B MINGW64 /d/ccc109/se/python/alg/03-monteCarlo/markov (master)
$ python mcmc.py
P(['b', 'a', 'b', 'b'])=0.06079

user@DESKTOP-96FRN6B MINGW64 /d/ccc109/se/python/alg/03-monteCarlo/markov (master)
$ python mcmc.py
P(['b', 'a', 'b', 'b'])=0.05979

user@DESKTOP-96FRN6B MINGW64 /d/ccc109/se/python/alg/03-monteCarlo/markov (master)
$ python mcmc.py
P(['b', 'a', 'b', 'b'])=0.05866
```

## gibbs.js


Gibbs Algorithm 的範例

問題：機率式有限狀態機，P(a=>b)=0.3, P(b=>a)=0.5   P(a=>b)=0.7, P(b=>b)=0.5

目標：尋找該「機率式有限狀態機」的穩態，也就是 P(a) = ?, P(b)=? 時系統會達到平衡。


```sh
user@DESKTOP-96FRN6B MINGW64 /d/ccc109/se/python/alg/03-monteCarlo/markov (master)
$ python gibbs.py
P1 =  :'a': 0.54, 'b': 0.46
P1 =  :'a': 0.608, 'b': 0.392
P1 =  :'a': 0.6215999999999999, 'b': 0.37839999999999996
P1 =  :'a': 0.62432, 'b': 0.37567999999999996
P1 =  :'a': 0.624864, 'b': 0.37513599999999997
標準答案:P(a)=5/8=0.625 P(b)=3/8=0.375
```

## mcmcGibbs.js

```sh
user@DESKTOP-96FRN6B MINGW64 /d/ccc109/se/python/alg/03-monteCarlo/markov (master)
$ python mcmcGibbs.py
P= :'a': 0.2, 'b': 0.8, 'a=>a': 0.7, 'a=>b': 0.3, 'b=>a': 0.5, 'b=>b': 0.5
P= :'a': 0.627399999999953, 'b': 0.3726000000000471, 'a=>a': 0.7, 'a=>b': 0.3, 'b=>a': 0.5, 'b=>b': 0.5
```