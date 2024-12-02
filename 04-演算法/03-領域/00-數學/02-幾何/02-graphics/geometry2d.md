# 二維圖形變換

## 旋轉

一個二維的旋轉矩陣可以表示為：

$$
\begin{bmatrix}
\cos{\theta} & -\sin{\theta} \\
\sin{\theta} & \cos{\theta}
\end{bmatrix}
$$

其中，$\theta$ 是旋轉的角度。例如，當 $\theta = \frac{\pi}{2}$ 時，旋轉矩陣為：

$$
\begin{bmatrix}
\cos{\frac{\pi}{2}} & -\sin{\frac{\pi}{2}} \\
\sin{\frac{\pi}{2}} & \cos{\frac{\pi}{2}}
\end{bmatrix}
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
$$

這個矩陣將向量 $(x, y)$ 旋轉 90 度，使其變為 $(-y, x)$。

## 縮放

$$
\begin{bmatrix}
s_x & 0 \\
0 & s_y
\end{bmatrix}
$$

其中，$s_x$ 和 $s_y$ 分別表示縮放在 x 和 y 方向上的比例。例如，如果我們希望將向量 $(x, y)$ 縮放 0.5 倍，則縮放矩陣為：

$$
\begin{bmatrix}
0.5 & 0 \\
0 & 0.5
\end{bmatrix}
$$

這個矩陣將向量 $(x, y)$ 縮放 0.5 倍，使其變為 $(0.5x, 0.5y)$。

## 平移

一個二維的平移矩陣可以表示為：

$$
\begin{bmatrix}
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1
\end{bmatrix}
$$

其中，$t_x$ 和 $t_y$ 分別表示平移的量在 x 和 y 方向上的比例。例如，如果我們希望將向量 $(x, y)$ 平移到 $(x+1, y+2)$，則平移矩陣為：

$$
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 2 \\
0 & 0 & 1
\end{bmatrix}
$$

這個矩陣將向量 $(x, y)$ 平移到 $(x+1, y+2)$。