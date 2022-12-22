# 微積分

## 微分

```
PS D:\ccc\ccc109a\se\deno\alg\09-numerical\calculus> deno run diff.js
diff(sin(x), pi/3) =  0.5000000413701855
cos(pi/3) =  0.5000000000000001
```

## 積分

```
PS D:\ccc\ccc109a\se\deno\alg\09-numerical\calculus> deno run integral.js
integral(sqrt, 0, 1) = 0.6614629471031478
integral(sin,  0, pi) = 1.9999900283082583
```

## 偏微分

```
PS D:\ccc\ccc109a\se\deno\alg\09-numerical\calculus> deno run partial.js
df(x:3,y:2)/dx= 6.009999999999849
df(x:3,y:2)/dy= 4.009999999999891
```


## reference

* https://docs.sympy.org/1.5.1/tutorial/calculus.html
    * 這個有 live 可以玩
