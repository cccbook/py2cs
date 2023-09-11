# n 維超球體的體積公式

$$
V_{n}={\pi ^{{\frac  {n}{2}}}R^{n} \over \Gamma ({\frac  {n}{2}}+1)}={C_{n}R^{n}}
$$

* [N維球面](https://zh.wikipedia.org/zh-tw/N%E7%BB%B4%E7%90%83%E9%9D%A2)

## ccc: 請問 n 維超球體的體積公式

ChatGPT:

n 維超球體的體積公式可以表示為：

V = (π^(n/2) / Γ(n/2 + 1)) * r^n

其中，V 是 n 維超球體的體積，π 是圓周率 (大約等於 3.14159)，Γ 是伽馬函數，r 是超球體的半徑。

請注意，這個公式需要使用伽馬函數，而計算伽馬函數可能需要特殊的數學函數庫或計算工具來進行。

## ndBall.py 執行結果

```
$ python ndBall.py
r= 1
ballVolume(2d, r)= 3.141592653589793
pi r^2= 3.141592653589793
mcBallVolume(2d, r)= 3.13336
ballVolume(3d, r)= 4.1887902047863905
4/3 pi r^3= 4.1887902047863905
mcBallVolume(3d, r)= 4.18944
ballVolume(4d, r)= 4.934802200544679
mcBallVolume(4d, r)= 4.93008
```