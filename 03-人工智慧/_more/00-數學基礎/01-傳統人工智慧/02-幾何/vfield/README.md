# 向量場 -- 梯度，散度，旋度

* [散度和旋度：馬克士威方程組中的符號以及流體等](https://www.youtube.com/watch?v=rB83DpBJQsE)

以下解釋請參考此文章。

* [圖解梯度、 散度與旋度(PDF)](https://web.math.sinica.edu.tw/math_media/d392/39203.pdf), 數學傳播 39卷2期, pp. 30-55, 林琦焜. (讚！)

另請參考紙本 kreyszig 工程數學精華本（歐亞書局） 8ed (7-10, 7-11) 

## 梯度

grad 是最斜的方向

grad(f,p) 就是函數 f 在 p 點上最斜的方向。

## 散度

以散度 divA 就是單位體積向量場 A 之通量 (flux)

散度 ⇐⇒ 通量(曲面積分) ⇐⇒ Gauss 散度定理

## 旋度

因為 |curl A · n| ≤ |curl A|, 所以旋度 curl A 可以詮釋為單位面積向量場 A 之最大環流。

因為線積分與座標無關, 所以旋度與所選取之座標無關,

旋度 ⇐⇒ 環流(線積分) ⇐⇒ Stokes定理

## Laplace 算子

∆ = div∇ = ∇ · ∇

## 線積分

```math
\int_C F(r) dr = \int_a^b F(r(t)) dr/dt dt
```

## 參考文獻

* [線代啟示錄 : 梯度、散度與旋度](https://ccjou.wordpress.com/2013/06/27/%E6%A2%AF%E5%BA%A6%E3%80%81%E6%97%8B%E5%BA%A6%E8%88%87%E6%95%A3%E5%BA%A6/)