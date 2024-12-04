import numpy as np
from sklearn.model_selection import train_test_split  # 導入訓練/測試集分割工具
from sklearn import datasets  # 導入數據集工具
import matplotlib.pyplot as plt  # 導入繪圖工具
from LinearRegression import LinearRegression  # 導入自定義的線性回歸模型

# 生成一個帶有噪聲的回歸數據集
X, y = datasets.make_regression(n_samples=100,  # 生成 100 個樣本
                                n_features=1,   # 每個樣本 1 個特徵
                                noise=20,       # 加入噪聲，噪聲程度為 20
                                random_state=4)  # 設定隨機種子，保證結果可重現

# 將數據集分割為訓練集和測試集，80% 為訓練集，20% 為測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 繪製數據的分佈圖，查看輸入數據的可視化
fig = plt.figure(figsize=(8,6))  # 設定圖的大小
plt.scatter(X[:, 0], y, color="b", marker="o", s=30)  # 用藍色散點圖繪製所有樣本點
plt.show()  # 顯示圖像

# 初始化線性回歸模型，學習率設為 0.01
reg = LinearRegression(lr=0.01)
# 使用訓練集數據進行模型訓練
reg.fit(X_train, y_train)
# 對測試集數據進行預測
predictions = reg.predict(X_test)

# 定義均方誤差（MSE）的計算函數
def mse(y_test, predictions):
    """
    計算均方誤差
    :param y_test: 測試集的實際值
    :param predictions: 測試集的預測值
    :return: 計算出的均方誤差
    """
    return np.mean((y_test - predictions) ** 2)

# 計算測試集的均方誤差
mse = mse(y_test, predictions)
print(mse)  # 輸出均方誤差

# 使用訓練好的模型對所有數據進行預測，繪製預測線
y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')  # 使用 'viridis' 調色盤
fig = plt.figure(figsize=(8,6))  # 設定圖像大小
# 用不同顏色標記訓練集和測試集
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)  # 用深色標記訓練集
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)  # 用淺色標記測試集
# 繪製預測的回歸線
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')  # 用黑色線繪製回歸線
plt.show()  # 顯示圖像
