https://chatgpt.com/c/675659cf-1010-8012-9f7e-c53db443310d

## ccc: 請給一些著名的控制系統範例，並給出該對應的方程式

以下是一些著名的控制系統範例及其對應的方程式：

---

### 1. **單擺（Simple Pendulum）**
#### 描述：
單擺是一個基本的非線性動力系統範例。若忽略阻尼及驅動力，系統是保守的。

#### 方程式：
\[
\frac{d^2\theta}{dt^2} + \frac{g}{L} \sin\theta = 0
\]
- \(\theta\)：擺角（弧度）
- \(g\)：重力加速度
- \(L\)：擺長

當角度小時，可以線性化為：
\[
\frac{d^2\theta}{dt^2} + \frac{g}{L} \theta = 0
\]

---

### 2. **質量-彈簧-阻尼系統（Mass-Spring-Damper System）**
#### 描述：
該系統描述了在彈簧恢復力和阻尼力作用下，質量的運動，常用於機械系統建模。

#### 方程式：
\[
m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = F(t)
\]
- \(m\)：質量
- \(c\)：阻尼係數
- \(k\)：彈簧常數
- \(x\)：質量的位移
- \(F(t)\)：外力

對應的狀態空間方程為：
\[
\dot{x}_1 = x_2, \quad \dot{x}_2 = -\frac{c}{m}x_2 - \frac{k}{m}x_1 + \frac{1}{m}F(t)
\]

---

### 3. **電路中的 RLC 串聯電路**
#### 描述：
RLC 串聯電路是控制電路分析的經典例子。

#### 方程式：
\[
L\frac{d^2q}{dt^2} + R\frac{dq}{dt} + \frac{q}{C} = V(t)
\]
- \(L\)：電感
- \(R\)：電阻
- \(C\)：電容
- \(q\)：電荷
- \(V(t)\)：電壓源

對應電流 \(i\) 的微分方程：
\[
L\frac{di}{dt} + Ri + \frac{1}{C} \int i \, dt = V(t)
\]

---

### 4. **伺服電機（DC Motor Control System）**
#### 描述：
直流伺服電機的動力方程用於模擬電機的角速度或位置控制。

#### 方程式：
機械部分：
\[
J\frac{d^2\theta}{dt^2} + b\frac{d\theta}{dt} = K_t i
\]
電氣部分：
\[
L\frac{di}{dt} + Ri = V - K_e\frac{d\theta}{dt}
\]
- \(J\)：轉動慣量
- \(b\)：阻尼係數
- \(K_t\)：轉矩常數
- \(K_e\)：反電動勢常數
- \(L\)：電感
- \(R\)：電阻
- \(i\)：電流
- \(V\)：電壓
- \(\theta\)：角位移

---

### 5. **倒立擺（Inverted Pendulum）**
#### 描述：
倒立擺是經典的非線性和不穩定控制系統，常用於驗證控制演算法。

#### 方程式：
\[
M\frac{d^2x}{dt^2} + b\frac{dx}{dt} - m\ell\frac{d^2\theta}{dt^2}\cos\theta + m\ell\left(\frac{d\theta}{dt}\right)^2\sin\theta = F
\]
\[
I\frac{d^2\theta}{dt^2} + m\ell\frac{d^2x}{dt^2}\cos\theta + m g \ell \sin\theta = 0
\]
- \(M\)：小車質量
- \(m\)：擺質量
- \(\ell\)：擺長
- \(I\)：擺慣量
- \(F\)：外力

---

### 6. **熱傳導系統（Thermal System）**
#### 描述：
熱傳導系統描述溫度隨時間的變化，常用於 HVAC 系統建模。

#### 方程式：
\[
C\frac{dT}{dt} + \frac{T - T_\text{ambient}}{R} = Q
\]
- \(C\)：熱容
- \(R\)：熱阻
- \(T\)：系統溫度
- \(T_\text{ambient}\)：環境溫度
- \(Q\)：加熱功率

---

這些範例涵蓋了機械系統、電氣系統和熱力系統的基本控制模型，對應的方程式可用於推導狀態空間或頻域表達式進行分析和設計。

## ccc: 請給出上述控制系統方程的 Python 求解程式

