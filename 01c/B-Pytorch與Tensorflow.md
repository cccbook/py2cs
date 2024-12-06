#### **附錄 B PyTorch 與 TensorFlow 的數學實現基礎**

本附錄旨在幫助讀者理解如何在 PyTorch 和 TensorFlow 等流行的深度學習框架中實現數學公式。這將涵蓋一些常見的數學操作，並展示如何將它們轉換為 PyTorch 或 TensorFlow 的程式碼實現，讓讀者能夠將理論與實際操作結合。

---

### **B.1 張量運算與基礎數學操作**

#### **PyTorch 實現**

在 PyTorch 中，張量（tensor）是其主要的數據結構，提供了對各種數學操作的支持，如加法、乘法、轉置等。

```python
import torch

# 創建張量
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 向量加法
c = a + b

# 點積
dot_product = torch.dot(a, b)

# 矩陣乘法
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
matrix_product = torch.mm(A, B)

# 張量轉置
A_T = A.T

# 張量的形狀
shape = A.shape
```

#### **TensorFlow 實現**

在 TensorFlow 中，數據結構稱為張量（tensor），並且也支持許多基本數學運算。

```python
import tensorflow as tf

# 創建張量
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

# 向量加法
c = a + b

# 點積
dot_product = tf.reduce_sum(a * b)

# 矩陣乘法
A = tf.constant([[1.0, 2.0], [3.0, 4.0]])
B = tf.constant([[5.0, 6.0], [7.0, 8.0]])
matrix_product = tf.matmul(A, B)

# 張量轉置
A_T = tf.transpose(A)

# 張量的形狀
shape = tf.shape(A)
```

---

### **B.2 激活函數**

激活函數在深度學習中起著至關重要的作用，常見的激活函數包括 ReLU、sigmoid 和 tanh。

#### **ReLU 激活函數**

- **PyTorch 實現：**

```python
relu = torch.nn.ReLU()
output = relu(a)
```

- **TensorFlow 實現：**

```python
output = tf.nn.relu(a)
```

#### **Sigmoid 激活函數**

- **PyTorch 實現：**

```python
sigmoid = torch.sigmoid(a)
```

- **TensorFlow 實現：**

```python
output = tf.sigmoid(a)
```

#### **Tanh 激活函數**

- **PyTorch 實現：**

```python
tanh = torch.tanh(a)
```

- **TensorFlow 實現：**

```python
output = tf.tanh(a)
```

---

### **B.3 自動微分與梯度計算**

自動微分是深度學習中的一個重要特性，它使得我們能夠在反向傳播中自動計算梯度。

#### **PyTorch 實現（Autograd）**

PyTorch 提供了自動微分功能，讓我們可以輕鬆地計算梯度。

```python
# 設定 requires_grad=True 以便追蹤梯度
x = torch.tensor(2.0, requires_grad=True)

# 定義函數
y = x**2 + 3*x + 1

# 反向傳播計算梯度
y.backward()

# 查看梯度
print(x.grad)  # Output: tensor(7.)
```

#### **TensorFlow 實現（Gradient Tape）**

TensorFlow 提供了 `tf.GradientTape` 用於記錄操作並計算梯度。

```python
# 設定變數以便求梯度
x = tf.Variable(2.0)

with tf.GradientTape() as tape:
    y = x**2 + 3*x + 1

# 計算梯度
dy_dx = tape.gradient(y, x)
print(dy_dx)  # Output: 7.0
```

---

### **B.4 梯度下降與優化**

在訓練模型時，梯度下降是最常用的優化算法。

#### **PyTorch 實現**

```python
# 創建優化器
optimizer = torch.optim.SGD([x], lr=0.01)

# 訓練過程
optimizer.zero_grad()  # 清除以前的梯度
y = x**2 + 3*x + 1
y.backward()  # 計算梯度
optimizer.step()  # 更新參數
```

#### **TensorFlow 實現**

```python
# 創建優化器
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# 訓練過程
with tf.GradientTape() as tape:
    y = x**2 + 3*x + 1
dy_dx = tape.gradient(y, x)
optimizer.apply_gradients([(dy_dx, x)])
```

---

### **B.5 損失函數**

損失函數在神經網絡訓練中用來衡量模型的預測結果與真實標籤之間的差距。

#### **均方誤差（MSE）**

- **PyTorch 實現：**

```python
loss_fn = torch.nn.MSELoss()
loss = loss_fn(predictions, targets)
```

- **TensorFlow 實現：**

```python
loss = tf.reduce_mean(tf.square(predictions - targets))
```

#### **交叉熵損失**

- **PyTorch 實現：**

```python
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(predictions, targets)
```

- **TensorFlow 實現：**

```python
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=predictions))
```

---

### **B.6 優化算法（Adam）**

Adam 是一種常用的優化算法，結合了動量法和自適應學習率。

#### **PyTorch 實現**

```python
optimizer = torch.optim.Adam([x], lr=0.001)

# 訓練過程
optimizer.zero_grad()
y = x**2 + 3*x + 1
y.backward()
optimizer.step()
```

#### **TensorFlow 實現**

```python
optimizer = tf.optimizers.Adam(learning_rate=0.001)

# 訓練過程
with tf.GradientTape() as tape:
    y = x**2 + 3*x + 1
dy_dx = tape.gradient(y, x)
optimizer.apply_gradients([(dy_dx, x)])
```

---

### **B.7 結論**

PyTorch 和 TensorFlow 都是深度學習領域中最流行的框架，兩者都提供了強大的數學運算能力，並且支持自動微分。理解這些數學操作如何在這些框架中實現，將幫助讀者深入理解語言模型的數學基礎並能夠有效地在實際應用中使用它們。