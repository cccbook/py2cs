#### **附錄 A 數學符號表與常用公式**

本附錄旨在整理本書中涉及的數學符號和常用公式，便於讀者快速查找和理解。以下是對書中數學概念的符號定義和公式總結：

---

### **A.1 向量與矩陣符號**

- 向量：  $\mathbf{v}, \mathbf{w}, \mathbf{x}$ 
- 矩陣：  $\mathbf{A}, \mathbf{B}, \mathbf{W}$ 
- 單位矩陣：  $\mathbf{I}$ 
- 轉置：  $\mathbf{A}^T$ 
- 逆矩陣：  $\mathbf{A}^{-1}$ 
- 迹（Trace）：  $\text{Tr}(\mathbf{A})$ 
- 行列式：  $\text{det}(\mathbf{A})$ 
- 向量點積：  $\mathbf{v} \cdot \mathbf{w}$ 
- 向量外積：  $\mathbf{v} \times \mathbf{w}$ 

---

### **A.2 張量運算**

- 張量：  $\mathbb{T}$ 
- 張量的轉置：  $\mathbb{T}^T$ 
- 張量積（Khatri-Rao積）：  $\mathbb{T}_1 \odot \mathbb{T}_2$ 
- 張量積：  $\mathbf{A} \otimes \mathbf{B}$ 
- 張量形狀：  $\text{shape}(\mathbb{T})$ 

---

### **A.3 函數與微積分符號**

- 函數：  $f(x), g(x)$ 
- 偏導數：  $\frac{\partial f}{\partial x}$ 
- 梯度：  $\nabla f(\mathbf{x})$ 
- 雅可比矩陣：  $\mathbf{J} = \frac{\partial f}{\partial \mathbf{x}}$ 
- 二階導數：  $\frac{\partial^2 f}{\partial x^2}$ 
- 梯度下降： 

```math
  \mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)

```
  其中， $\eta$  是學習率， $\nabla f(\mathbf{x}_t)$  是梯度。

---

### **A.4 概率與統計符號**

- 概率密度函數：  $p(x)$ 
- 條件概率：  $p(x|y)$ 
- 貝葉斯定理：

```math
  p(x|y) = \frac{p(y|x)p(x)}{p(y)}

```
- 高斯分布：  $\mathcal{N}(\mu, \sigma^2)$ 
- 均值：  $\mu = \frac{1}{N} \sum_{i=1}^N x_i$ 
- 方差：  $\sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2$ 
- 區間估計：  $[\hat{\mu} - Z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{N}}, \hat{\mu} + Z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{N}}]$ 

---

### **A.5 信息論符號與公式**

- 熵：  $H(X) = -\sum_{i} p(x_i) \log p(x_i)$ 
- 交叉熵： 

```math
  H(p, q) = -\sum_{i} p(x_i) \log q(x_i)

```
- Kullback-Leibler散度（KL散度）： 

```math
  D_{\text{KL}}(p || q) = \sum_{i} p(x_i) \log \frac{p(x_i)}{q(x_i)}

```
- 信息增益： 

```math
  \text{IG}(X) = H(X) - H(X|Y)

```

---

### **A.6 生成模型符號**

- 概率分布：  $p(x)$ 
- 梯度上升： 

```math
  \theta_{t+1} = \theta_t + \eta \nabla_{\theta} \log p(x | \theta)

```
- 最大似然估計（MLE）： 

```math
  \hat{\theta}_{MLE} = \arg \max_{\theta} \prod_{i=1}^N p(x_i | \theta)

```
- 最大後驗估計（MAP）： 

```math
  \hat{\theta}_{MAP} = \arg \max_{\theta} p(\theta) \prod_{i=1}^N p(x_i | \theta)

```

---

### **A.7 深度學習與神經網絡符號**

- 神經元輸出：  $y = \sigma(\mathbf{w}^T \mathbf{x} + b)$ 
- 激活函數：  $\sigma(x)$ （如ReLU、sigmoid、tanh）
- 損失函數：  $L = \frac{1}{N} \sum_{i=1}^N \text{Loss}(y_i, \hat{y}_i)$ 
- 反向傳播： 

```math
  \frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{w}}

```

---

### **A.8 Transformer模型符號**

- 自注意力： 

```math
  \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V

```
- 多頭注意力： 

```math
  \text{MultiHead}(Q, K, V) = \text{concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O

```
  其中， $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ 
- 位置編碼： 

```math
  PE(t, 2i) = \sin(t / 10000^{2i/d})
  \quad PE(t, 2i+1) = \cos(t / 10000^{2i/d})

```

---

### **A.9 優化算法符號**

- 梯度下降： 

```math
  \theta_{t+1} = \theta_t - \eta \nabla_{\theta} J(\theta)

```
- Adam算法： 

```math
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_{\theta} J(\theta)

```

```math
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) \nabla_{\theta} J(\theta)^2

```

```math
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}

```

```math
  \theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

```

---

### **A.10 微積分與數學分析公式**

- 鏈式法則： 

```math
  \frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)

```
- 泰勒展開： 

```math
  f(x) \approx f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \dots

```

---

這些符號和公式在本書中涵蓋了大部分的數學基礎，對於學習和理解大規模語言模型的數學背景非常重要。