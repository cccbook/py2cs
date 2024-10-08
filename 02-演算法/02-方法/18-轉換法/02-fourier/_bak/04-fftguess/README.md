# fftguess

從結果解回 fft(cos(qx)) 的 q 到底是多少。

答案：不是零的那個索引值就是 q 了。

原因：

```math
a0 = 1/2L \int_:a^:b f(x) dx \\

an = 1/L \int_:a^:b f(x) cos(n \Pi x/L) dx \\

bn = 1/L \int_:a^:b f(x) sin(n \Pi x/L) dx \\
```
