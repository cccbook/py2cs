# fft2L

用圖形顯示 cos(at) (a=0, 1, N/2, N) 轉為傅立葉級數之後，哪個係數會不為零。

注意：任意週期 p=2L 的傅立葉級數，可由標準的公式乘上 pi/L 縮放獲得。 

（參考高等工程數學 KREYSIG 8th ed 中文版 498 頁，公式 2）

```math
a0 = 1/2L \int_:-L^:L f(x) dx \\

an = 1/L \int_:-L^:L f(x) cos(n \Pi x/L) dx \\

bn = 1/L \int_:-L^:L f(x) sin(n \Pi x/L) dx \\
```

比較：（以下為標準版）

```math
a0 = 1/2\pi \int_:-\pi^:pi f(x) dx \\

an = 1/2\pi \int_:-\pi^:pi f(x) cos(nx) dx \\

bn = 1/2\pi \int_:-\pi^:pi f(x) sin(nx) dx \\
```

