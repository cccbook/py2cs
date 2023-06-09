## 附錄A：量子力學中的數學工具

在量子力學中，數學是不可或缺的。下面是一些量子力學中常見的數學工具:

1. 薛定譯方程 (Schrödinger equation)：薛定譯方程是量子力學描述量子系統時間演化的基本方程。它可以用數學公式表示為：

$$i\hbar\frac{\partial}{\partial t}\Psi(\vec{r}, t) = \hat{H}\Psi(\vec{r}, t)$$

其中， $i$是虛數單位， $\hbar$是約化 Planck 常數， $\Psi(\vec{r}, t)$是波函數， $\hat{H}$是哈密頓算符。

2. 哈密頓算符 (Hamiltonian operator)：哈密頓算符描述了量子系統的能量。在薛定譯方程中，哈密頓算符作為演化算符，它可以用數學公式表示為：

$$\hat{H} = -\frac{\hbar^2}{2m}\vec{\nabla}^2 + V(\vec{r})$$

其中， $m$是粒子的質量， $\vec{\nabla}^2$是拉普拉斯算符， $V(\vec{r})$是位能函數。

3. 波函數 (Wave function)：波函數是表示量子系統狀態的數學函數。波函數的平方正比於量子系統在某一位置出現的概率。它可以用數學公式表示為：

$$\Psi(\vec{r}, t) = \psi(\vec{r})e^{-iEt/\hbar}$$

其中， $\psi(\vec{r})$是空間部分， $e^{-iEt/\hbar}$是時間部分， $E$是能量。

4. 自旋 (Spin)：自旋是粒子的一種內禀角動量。它可以用一個算符來描述：

$$\hat{S} = \frac{\hbar}{2}\vec{\sigma}$$

其中， $\vec{\sigma}$是 Pauli 矩陣向量，表示粒子的自旋在三個方向上的投影。

5. 期望值 (Expectation value)：量子系統某一物理量的期望值是統計平均值。在波函數表示下，物理量的期望值可以表示為：

$$\langle\hat{A}\rangle = \int\Psi^*(\vec{r}, t)\hat{A}\Psi(\vec{r}, t)d^3\vec{r}$$

其中， $\hat{A}$是物理量算符， $d^3\vec{r}$表示空間積分。

6. 系統的演化 (Evolution of a system)：例如，如果一個系統在初始時間 $t_0$ 的波函數為 $\Psi(\vec{r}, t_0)$，在 $t>t_0$ 的波函數可以通過演化算符 $\hat{U}(t,t_0)$ 表示：

$$\Psi(\vec{r}, t) = \hat{U}(t,t_0)\Psi(\vec{r}, t_0)$$

其中， $\hat{U}(t,t_0)$ 是演化算符，可以表示為：

$$\hat{U}(t,t_0) = e^{-\frac{i}{\hbar}\hat{H}(t-t_0)}$$

這裡的 $\hat{H}$ 是哈密頓算符。

以上是量子力學中比較常見的數學工具，它們在量子力學的理論和應用研究中起著至關重要的作用。