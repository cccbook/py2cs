# logicmin

* https://github.com/dreylago/logicmin

LogicMin is a Python package that minimize boolean functions using the tabular method of minimization (Quine-McCluskey).

## fulladder.py

```
$ python fulladder.py
Co <= a.b + Ci.b + Ci.a
s <= Ci'.a'.b + Ci'.a.b' + Ci.a'.b' + Ci.a.b
Co: -----------------------
Cost minimal: 17.0. Canonical: 35.0 (-51.4%)
Comps: 9. Iterations: 1. Minimal: True
s: -----------------------
Cost minimal: 26.0. Canonical: 25.0 (4.0%)
Comps: 6. Iterations: 1. Minimal: True
$
```

## bcd27seg.py

```
$ python bcdto7seg.py 
g <= b2'.b1 + b2.b1' + b2.b0' + b3 
f <= b1'.b0' + b2.b1' + b2.b0' + b3
e <= b2'.b0' + b1.b0'
d <= b2.b1'.b0 + b2'.b0' + b2'.b1 + b1.b0'
c <= b1' + b0 + b2
b <= b1'.b0' + b1.b0 + b2'
a <= b2'.b0' + b1.b0 + b2.b0 + b3
```

## binaryCounter.py

```
$ python binaryCounter.py
Y <= Q0
D0 <= X.Q0' + X'.Q0
D1 <= X.Q1'.Q0 + Q1.Q0' + X'.Q1
```
