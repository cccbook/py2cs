### **附錄 A3 - Python 在連續數學中的應用**

Python 是當今最流行的編程語言之一，特別是在科學計算和數學建模領域。Python 擁有強大的數學庫，如 `NumPy`、`SciPy`、`SymPy` 和 `matplotlib`，使得它成為進行數值運算、符號計算以及數學可視化的理想工具。本附錄介紹了 Python 在連續數學中的一些應用，包括數值積分、微分方程求解、傅立葉變換等。

---

#### **1. 數值積分**
數值積分是數學中處理積分問題的一種方法，特別是在無法解析求解的情況下。Python 提供了 `SciPy` 庫中的 `integrate` 模塊來進行數值積分。

**範例：計算積分**

```python
import scipy.integrate as integrate

# 定義被積分的函數
def f(x):
    return x**2

# 計算積分
result, error = integrate.quad(f, 0, 1)
print(f"積分結果: {result}, 誤差估計: {error}")
```

這段程式碼計算了從 0 到 1 的積分 \( \int_0^1 x^2 \, dx \)，結果為 1/3。

---

#### **2. 微分方程求解**
Python 的 `SciPy` 庫也包含了求解常微分方程（ODEs）的一些功能。使用 `odeint` 函數，可以方便地求解初值問題。

**範例：求解簡單的微分方程**

```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定義微分方程
def model(y, t):
    dydt = -2 * y
    return dydt

# 初始條件
y0 = 1
t = np.linspace(0, 5, 100)

# 求解微分方程
y = odeint(model, y0, t)

# 畫圖
plt.plot(t, y)
plt.xlabel('時間 t')
plt.ylabel('y(t)')
plt.title('微分方程的解: y\' = -2y')
plt.show()
```

這段程式碼求解了常微分方程 \( y'(t) = -2y(t) \)，並繪製了解的圖形。可以看到，解隨時間指數衰減。

---

#### **3. 符號計算：微分與積分**
在進行符號計算（即不進行數值逼近，而是解析計算）時，`SymPy` 是一個強大的 Python 库。它可以進行微分、積分、極限、級數展開等數學操作。

**範例：符號積分與微分**

```python
import sympy as sp

# 定義符號變數
x = sp.symbols('x')

# 定義函數
f = sp.sin(x)**2

# 計算導數
f_prime = sp.diff(f, x)
print(f"導數: {f_prime}")

# 計算積分
f_integral = sp.integrate(f, x)
print(f"積分: {f_integral}")
```

這段程式碼定義了函數 \( \sin^2(x) \)，並計算了它的導數和積分。

---

#### **4. 傅立葉變換**
傅立葉變換在信號處理和數學分析中具有重要應用，Python 可以用來計算連續和離散的傅立葉變換。`NumPy` 提供了快速傅立葉變換（FFT）算法，而 `scipy.fft` 模塊則提供了更多傅立葉變換工具。

**範例：計算離散傅立葉變換**

```python
import numpy as np
import matplotlib.pyplot as plt

# 定義時間變數
t = np.linspace(0, 1, 500, endpoint=False)

# 定義信號（例如正弦波）
signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# 計算離散傅立葉變換
fft_signal = np.fft.fft(signal)

# 計算頻率
frequencies = np.fft.fftfreq(len(t), t[1] - t[0])

# 顯示頻譜圖
plt.plot(frequencies, np.abs(fft_signal))
plt.xlabel('頻率 (Hz)')
plt.ylabel('幅度')
plt.title('傅立葉變換頻譜')
plt.show()
```

這段程式碼計算了信號的傅立葉變換並繪製了其頻譜圖，可以看到信號的頻率成分。

---

#### **5. 解線性微分方程組**
對於線性微分方程組，Python 的 `scipy.linalg` 和 `odeint` 等工具可以用來進行求解。以下範例展示了如何使用 `odeint` 解決一個簡單的線性微分方程組。

**範例：求解線性微分方程組**

```python
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# 定義線性微分方程組
def system(y, t):
    x, v = y
    dxdt = v
    dvdt = -9.81  # 假設重力加速度為9.81 m/s^2
    return [dxdt, dvdt]

# 初始條件：位置x=0, 速度v=0
y0 = [0, 0]

# 時間區間
t = np.linspace(0, 10, 100)

# 求解微分方程組
solution = odeint(system, y0, t)

# 繪製結果
plt.plot(t, solution[:, 0], label="位置 x(t)")
plt.plot(t, solution[:, 1], label="速度 v(t)")
plt.xlabel('時間 (s)')
plt.ylabel('值')
plt.title('簡單的線性微分方程組解')
plt.legend()
plt.show()
```

這段程式碼解了一個描述物體自由下落的簡單物理模型，其中位置 \( x(t) \) 和速度 \( v(t) \) 由一個線性微分方程組描述。

---

#### **6. 數值優化與最小化**
數值優化方法常用於最小化某個函數，Python 中的 `scipy.optimize` 模塊提供了多種方法來進行優化計算。這在科學計算、機器學習和信號處理中非常有用。

**範例：最小化函數**

```python
from scipy.optimize import minimize

# 定義一個要最小化的函數
def objective(x):
    return x**2 + 4*x + 4

# 執行優化
result = minimize(objective, 0)  # 初始猜測為 0
print(f"最小值: {result.fun}, 最小點: {result.x}")
```

這段程式碼最小化了函數 \( f(x) = x^2 + 4x + 4 \)，並打印出最小值和最小點。

---

#### **總結**
Python 作為數學與科學計算的強大工具，結合了多種數學庫的功能，能夠有效處理各種連續數學問題。無論是數值積分、微分方程求解、傅立葉變換還是符號計算，Python 都能夠提供簡單高效的解決方案。這使得 Python 成為數學家、工程師和科學家進行數學建模和分析的首選語言之一。