## A.3 微分方程和分析方法

## 微分方程基本概念

微分方程（Differential Equation，DE）是用一元函数及其导数表示的等式。通常含有未知函数及其导数, 而方程右边常常含有自变量.其一般形式为：

$$ F(x,y,y',y'',\cdots,y^{(n)}) = 0 $$

其中 $x$ 为自变量， $y$ 为未知函数， $y', y'', \cdots, y^{(n)}$ 分别为 $y$ 的一到 $n$ 阶导数，而 $F$ 是已知函数。若 $F$ 中不含自变量 $x$，则称微分方程为常微分方程（Ordinary Differential Equation，ODE），否则为偏微分方程（Partial Differential Equation，PDE）。

微分方程的解是指满足微分方程的函数，解的求解通常可以利用一些方法来简化和求解微分方程。

### 常微分方程

当方程只含有关于一自变量的函数及其一阶和（或）高阶导数时，即为常微分方程，一般形式为

$$ F(x, y, y’, y’’, ..., y^{(n)}) = 0 $$

其中 $y$ 是自变量 $x$ 的函数， $y’$ 是 $y$ 的一阶导数， $y’’$ 是 $y$ 的二阶导数， $n$ 是整数。

### 偏微分方程

当方程中含有多个独立变量，以及通过这些变量对未知函 数的各种偏导数，就称这个方程是偏微分方程。例如：

$$ \frac{\partial}{\partial x} u(x, y) + \frac{\partial}{\partial y} u(x, y) = 0 $$

其中 $u(x, y)$ 表示未知函数 $f$ 在二维平面上的取值，它的数学定义为：

$$ u(x,y)=f(x,y) $$

### 常数

在微分方程中常常引入常数，其通常有两种含义：

1. 常数是指微分方程的一个参数，等价于初值或边界条件，通过常数可以得到微分方程的具体解；
2. 构造常数的过程叫做积分常数的引入，即通过对微分方程连续求导直到函数表达式中只剩下未知函数 $y$ 的情况，然后再对其进行积分的过程。在积分过程中，由于导数与原函数之间存在着积分关系，因此每次积分都会多出一个常数。

## 常微分方程

### 一阶方程

#### 可分离变量的一阶方程

形如

$$ \frac{\mathrm{d}y}{\mathrm{d}x} = f(x)g(y) $$

的微分方程称为可分离变量的一阶方程。对其进行变量分离得到

$$ \frac{\mathrm{d}y}{g(y)} = f(x)\mathrm{d}x $$

对两边同时积分，即可得到 $y$ 关于 $x$ 的通解：

$$ \int \frac{\mathrm{d}y}{g(y)} = \int f(x)\mathrm{d}x + C$$

其中 $C$ 为常数.

#### 齐次方程

形如

$$ \frac{\mathrm{d}y}{\mathrm{d}x} = F\left(\frac{y}{x}\right) $$

的微分方程称为齐次方程。对其进行变量替换 $y = vx$，即可化为可分离变量的一阶方程，即

$$\begin{aligned} \frac{\mathrm{d}y}{\mathrm{d}x} &= \frac{\mathrm{d}vx}{\mathrm{d}x} = v + x\frac{\mathrm{d}v}{\mathrm{d}x} \\ &= F\left(\frac{v}{x}\right) = F\left(\frac{y}{x}\right) \end{aligned}$$

解得 

$$\int \frac{\mathrm{d}v}{F(v) - v} = \int \frac{\mathrm{d}x}{x} + C$$

该方程的通解为 $y = vx$。

#### 一阶线性方程

形如

$$ \frac{\mathrm{d}y}{\mathrm{d}x} + p(x)y = q(x) $$

的微分方程称为一阶的线性微分方程。
设 $\mu(x)$ 为其积分系数，两边同乘以 $\mu(x)$，可得

$$\begin{aligned} \mu(x)\frac{\mathrm{d}y}{\mathrm{d}x} + \mu(x)p(x)y &= \mu(x)q(x) \\ \frac{\mathrm{d}}{\mathrm{d}x}\left[\mu(x)y\right] &= \mu(x)q(x) \end{aligned}$$

解得

$$ y = \frac{1}{\mu(x)}\left[\int\mu(x)q(x)\mathrm{d}x + C\right] $$


#### 一阶非线性方程

形如

$$ \frac{\mathrm{d}y}{\mathrm{d}x} = f(x,y) $$

的微分方程称为一阶的非线性微分方程。

#### Bernoulli 方程

形如

$$ \frac{\mathrm{d}y}{\mathrm{d}x} + p(x)y = q(x)y^\alpha $$

的微分方程称为 Bernoulli 方程。对其进行变量替换 $z = y^{1-\alpha}$，即可化为线性微分方程。

#### Riccati 方程

形如

$$ \frac{\mathrm{d}y}{\mathrm{d}x} = p(x)y^2 + q(x)y + r(x) $$

的微分方程称为 Riccati 方程，一些 Riccati 方程是可简化为二阶线性微分方程的，例如 $y’=ay^2 + by +c$。

### 二阶方程

#### 二阶常系数齐次线性微分方程

形式为

$$ y’’ + py’ + qy = 0 $$

其中 $p,q$ 均为常数，称为二阶常系数齐次线性微分方程。

当 $p^2 - 4q < 0$ 时，称为无震荡系统。解称为解析解，可以表示成振幅和相位角的形式，即

$$ y = C_1e^{\alpha x}\cos \beta x + C_2e^{\alpha x}\sin \beta x $$

其中， $\alpha=-\frac{p}{2}$, $\beta=\frac{\sqrt{4q-p^{2}}}{2}$ 是两个常数.

当 $p^2 - 4q > 0$ 时，称为有震荡系统。解称为稳定解，可以表示成指数形式，即

$$ y = e^{\lambda_1 x}(C_1 \cos \mu x + C_2 \sin \mu x) + e^{\lambda_2 x}(C_3 \cos \nu x + C_4\sin \nu x) $$

其中 $\lambda_1, \lambda_2$ 是两个不同的实数， $\mu, \nu$ 是非零实数。 $C_1, C_2, C_3, C_4$ 都是常数。

当 $p^2 - 4q = 0$ 时，称为临界系统。通解可以表示为

$$ y = (C_1 + C_2x)e^{-\frac{p}{2}x} $$

#### 二阶常系数非齐次线性微分方程

形式为

$$ y’’ + py’ + qy = f(x) $$

其中 $p$, $q$ 是常数， $f(x)$ 是已知函数，称为二阶常系数非齐次线性微分方程。

设 $y = y_h + y_p$，则

1. 关于齐次方程 $y’’ + py’ + qy = 0$ 的解为 $y_h = C_1e^{\alpha x}\cos \beta x + C_2e^{\alpha x}\sin \beta x$，其中 $\alpha=-\frac{p}{2}$, $\beta=\frac{\sqrt{4q-p^{2}}}{2}$ 是两个常数.
2. 非齐次方程的一个特解 $y_p$ 取决于 $f(x)$ 的形式，可通过常数变易法求解，特别地，当 $f(x)$ 为 $e^{ax}$， $\cos b x$， $\sin b x$， $x^n e^{ax}$ 的形式时，可以直接取 $y_p$ 的形式.

## 常用数学物理方法

### 变分法

变分法是一种优雅而强大的数学方法，它可以用来求函数的极大值、极小值，或求解微积分方程的解。

其基本思想是在不确定函数值的情况下寻找最值。因此，对于确定性函数，转化为极值问题就成了优化问题。

形式上，考虑函数 $y(x)$ 和其在 $x_1$和 $x_2$ 两点的值 $y_1$ 和 $y_2$。则求函数 $y(x)$ 的极小值（或者极大值）等价于求解如下泛函的极小值（或者极大值）：

$$ J[y] = \int_{x_1}^{x_2}F(x,y,y')\mathrm{d}x $$

其中 $y’=\frac{\mathrm{d}y}{\mathrm{d}x}$，且积分界和 $y_1,y_2$ 是指定的。

### 特殊函数

在数学和物理领域，有很多特殊函数，这些函数在某些公式、方程的求解中具有很重要的作用。以下列举一些常见的特殊函数。

#### 贝塞尔函数

贝塞尔函数（Bessel Function）是用于解决圆对称问题的基本特殊函数。

第一类贝塞尔函数：

$$ J_n(x) = \sum_{m=0}^\infty \frac{(-1)^m}{m!(m+n)!} \left( \frac{x}{2} \right)^{2m+n} $$

第二类贝塞尔函数：

$$ Y_n(x) = \frac{J_n(x)\cos(n\pi)-J_{-n}(x)}{\sin(n\pi)} $$

#### 阶乘函数

阶乘函数是一种特殊的函数，它定义如下：

$$ n! = \prod_{i=1}^n i $$

其中 $n$ 为正整数。

#### 伽玛函数

伽玛函数（Gamma Function）是一种扩展了阶乘到复数和更一般的值域上的一类特殊函数。它有许多与阶乘有关的基本性质，并且其变体与各种数学领域，特别是函数分析和组合数学有着直接的联系。

$$ \Gamma(z) = \int_0^{\infty} x^{z-1}e^{-x} \mathrm{d}x $$

其中 $z$ 是复数。

#### 超几何函数

超几何函数是解决物理和数学学科中许多方程和问题的特殊函数。其一般形式定义如下：

$$ _2F_1(a,b;c;z) = \sum_{n=0}^\infty \frac{(a)_n(b)_n}{(c)_n}\frac{z^n}{n!} $$

其中 $(a)_n$ 为 Pochhammer 符号，定义为

$$ (a)_n = \begin{cases} a(a+1)(a+2)\cdots(a+n-1), &n\in\mathbb{N}^{*} \\ 1,&n=0 \end{cases}$$

此外，它还满足下列恒等式：

$$ _2F_1(a,b;c;1)= \frac{\Gamma(c)\Gamma(c-a-b)}{\Gamma(c-a)\Gamma(c-b)} $$

### 物理数学方法

物理数学方法是用数学手段研究物理问题的方法，背后涵盖了很多高深而又有趣的数学知识和技巧。以下列举一些经典的物理数学方法。

#### 分离变量法

分离变量法是一种用于解决偏微分方程的方法。其基本思想是将函数表示为多个函数的乘积形式，然后将每个函数的变量分离开来，逐一求解。以一般的二维波动方程为例，其可以表示为

$$\frac{\partial^{2} u}{\partial t^{2}}=c^{2}\left(\frac{\partial^{2} u}{\partial x^{2}}+\frac{\partial^{2} u}{\partial y^{2}}\right)$$

对其进行变量分离

$$u(x,y,t)=X(x) Y(y) T(t)$$

将其带回原方程，得到

$$\frac{T^{\prime\prime}(t)}{c^{2} T(t)}=\frac{X^{\prime\prime}(x)}{X(x)}+\frac{Y^{\prime\prime}(y)}{Y(y)}=\lambda$$

然后分别解出 $X(x)$， $Y(y)$ 和 $T(t)$，最后得到原方程的解。

#### 特征线法

特征线法是求解偏微分方程的一种方法，它的特点是使用曲线的参数方程描述偏微分方程的变化。通常使用变换将偏微分方程的表示变为只包含一阶导数的形式，然后构造微分方程的特征曲线，并沿着特征曲线解微分方程。以一般的二维波动方程为例，假设 $u(x,y,t)$ 为方程的解，其特征曲线的方程为

$$\frac{d y}{d x}=\frac{\partial u / \partial t}{\partial u / \partial x}$$

将其带回原方程，得到特征方程

$$\frac{\partial^{2} u}{\partial t^{2}}=c^{2}\left(\frac{\partial^{2} u}{\partial x^{2}}+\frac{\partial^{2} u}{\partial y^{2}}\right)$$

通过解特征方程，可得到特征曲线方程，进而求解出偏微分方程的解。

#### 变换法

变换法是一种将复杂的数学问题通过特定的变换转化为另一类比较容易处理的问题的方法。其中，奇异变换是一种常用的变换之一，它的特点是将微积分方程中包含奇异点的函数变换为自变量中含有极点的形式。以一般的二阶线性微分方程为例，设其形式为

$$y^{\prime \prime}(x)+p(x) y^{\prime}(x)+q(x) y(x)=g(x)$$

设 $\tau$ 是自变量 $x$ 的一个新函数，满足 $x=x(\tau)$。将 $y(x)$ 和 $x$