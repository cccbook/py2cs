# 三維圖學

## 三維平移矩陣

三維平移矩陣可以表示為：

$$
\begin{bmatrix}
1 & 0 & 0 & t_x \\
0 & 1 & 0 & t_y \\
0 & 0 & 1 & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

其中，$t_x$，$t_y$ 和 $t_z$ 分別表示平移的量在 x，y 和 z 方向上的比例。例如，如果我們希望將向量 $(x, y, z)$ 平移到 $(x+1, y+2, z+3)$，則平移矩陣為：

$$
\begin{bmatrix}
1 & 0 & 0 & 1 \\
0 & 1 & 0 & 2 \\
0 & 0 & 1 & 3 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

這個矩陣將向量 $(x, y, z)$ 平移到 $(x+1, y+2, z+3)$。

## 三維縮放矩陣

三維縮放矩陣可以表示為：

$$
\begin{bmatrix}
s_x & 0 & 0 \\
0 & s_y & 0 \\
0 & 0 & s_z
\end{bmatrix}
$$

其中，$s_x$，$s_y$ 和 $s_z$ 分別表示縮放在 x，y 和 z 方向上的比例。例如，如果我們希望將向量 $(x, y, z)$ 縮放 0.5 倍，則縮放矩陣為：

$$
\begin{bmatrix}
0.5 & 0 & 0 \\
0 & 0.5 & 0 \\
0 & 0 & 0.5
\end{bmatrix}
$$

這個矩陣將向量 $(x, y, z)$ 縮放 0.5 倍，使其變為 $(0.5x, 0.5y, 0.5z)$。

## 三維旋轉矩陣

三維旋轉矩陣可以表示

為：

$$
\begin{bmatrix}
\cos{\theta_x} & -\sin{\theta_x} & 0 \\
\sin{\theta_x} & \cos{\theta_x} & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos{\theta_y} & -\sin{\theta_y} \\
0 & \sin{\theta_y} & \cos{\theta_y}
\end{bmatrix}
\begin{bmatrix}
\cos{\theta_z} & 0 & -\sin{\theta_z} \\
0 & 1 & 0 \\
\sin{\theta_z} & 0 & \cos{\theta_z}
\end{bmatrix}
$$

其中，$\theta_x$，$\theta_y$ 和 $\theta_z$ 分別表示旋轉的角度在 x，y 和 z 方向上的比例。例如，如果我們希望將向量 $(x, y, z)$ 旋轉 90 度，則旋轉矩陣為：

$$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & -1 \\
0 & 1 & 0
\end{bmatrix}
$$

這個矩陣將向量 $(x, y, z)$ 旋轉 90 度，使其變為 $(-z, y, x)$。

注意，上面的矩陣並不是唯一的三維旋轉矩陣的形式。只要滿足旋轉矩陣滿足矩陣乘法的結合律，就可以把旋轵分為多個二維旋轉矩陣。例如，如果我們希望將向量 $(x, y, z)$ 旋轉 90 度，則旋轉矩陣也可以表示為：

