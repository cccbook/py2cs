import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# 生成一些簡單的回歸數據
np.random.seed(42)
X = np.linspace(-5, 5, 100)
y = X * 2 + 1 + np.random.normal(0, 1, X.shape)

# 將數據轉換為TensorFlow格式
X_train = tf.convert_to_tensor(X, dtype=tf.float32)
y_train = tf.convert_to_tensor(y, dtype=tf.float32)

# 定義貝氏神經網絡層
class BayesianDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None):
        super(BayesianDense, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.kernel_posterior = tfp.layers.default_mean_field_normal_fn()
        self.bias_posterior = tfp.layers.default_mean_field_normal_fn()
        self.dense = tfp.layers.DenseFlipout(self.units, activation=self.activation)

    def call(self, inputs):
        return self.dense(inputs)

# 定義貝氏神經網絡模型
model = tf.keras.Sequential([
    BayesianDense(64, activation='relu'),
    BayesianDense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 定義損失函數和優化器
model.compile(optimizer='adam', loss='mean_squared_error')

# 訓練模型
model.fit(X_train, y_train, epochs=200, batch_size=32)

# 進行預測
X_test = np.linspace(-5, 5, 100)
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_pred = model(X_test_tensor)

# 進行貝氏推理：獲取預測分佈
y_pred_mean = tf.reduce_mean(y_pred, axis=0)
y_pred_stddev = tf.math.reduce_std(y_pred, axis=0)

# 畫出結果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Training data', color='blue')
plt.plot(X_test, y_pred_mean, label='Predicted mean', color='red')
plt.fill_between(X_test, y_pred_mean - 2 * y_pred_stddev, y_pred_mean + 2 * y_pred_stddev,
                 color='red', alpha=0.2, label='Uncertainty (95% CI)')
plt.legend()
plt.title('Bayesian Neural Network for Regression')
plt.show()
