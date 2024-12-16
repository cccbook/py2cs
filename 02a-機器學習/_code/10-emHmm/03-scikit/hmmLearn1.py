from hmmlearn.hmm import GaussianHMM
import numpy as np
import matplotlib.pyplot as plt

# 生成數據
np.random.seed(42)
n_samples = 1000
# 兩個高斯分佈的觀察數據
X = np.concatenate([np.random.normal(0, 1, (n_samples // 2, 1)),
                   np.random.normal(5, 1, (n_samples // 2, 1))])

# 創建 HMM 模型
hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)

# 擬合模型
hmm.fit(X)

# 預測狀態序列
hidden_states = hmm.predict(X)

# 顯示結果
plt.figure(figsize=(15, 8))
plt.subplot(211)
plt.title("Hidden Markov Model - Observations")
plt.plot(X)
plt.subplot(212)
plt.title("Hidden States Sequence")
plt.plot(hidden_states)
plt.show()

# 查看 HMM 的參數
print("Means:\n", hmm.means_)
print("Covariances:\n", hmm.covars_)
