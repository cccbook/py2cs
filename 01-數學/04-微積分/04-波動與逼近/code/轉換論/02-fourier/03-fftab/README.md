# fftab

用圖形顯示 cos(at) (a=0, 1, N/2, N) 轉為傅立葉級數之後，哪個係數會不為零。

注意：(a,b) 區間的傅立葉級數，可由 p=2L 的傅立葉級數，位移到 (a+b)/2 獲得。

```math
a0 = 1/2L \int_:a^:b f(x) dx \\

an = 1/L \int_:a^:b f(x) cos(n \Pi x/L) dx \\

bn = 1/L \int_:a^:b f(x) sin(n \Pi x/L) dx \\
```

比較：（以下為標準版）

```math
a0 = 1/2\pi \int_:-\pi^:pi f(x) dx \\

an = 1/2\pi \int_:-\pi^:pi f(x) cos(nx) dx \\

bn = 1/2\pi \int_:-\pi^:pi f(x) sin(nx) dx \\
```

