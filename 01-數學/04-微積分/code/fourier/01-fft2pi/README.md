# fft2pi

用圖形顯示 cos(at) (a=0, 1, N/2, N) 轉為傅立葉級數之後，哪個係數會不為零。

注意：標準傅立葉級數的範圍是從 -pi 到 pi, 區分為 2N 份。（其中最高頻週期為 N）

（參考高等工程數學 KREYSIG 8th ed 中文版 492 頁，公式 6）

```math
a0 = 1/2\pi \int_:-\pi^:pi f(x) dx \\

an = 1/2\pi \int_:-\pi^:pi f(x) cos(nx) dx \\

bn = 1/2\pi \int_:-\pi^:pi f(x) sin(nx) dx \\
```
