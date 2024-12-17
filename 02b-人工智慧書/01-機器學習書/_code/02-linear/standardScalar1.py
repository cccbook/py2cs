from sklearn.preprocessing import StandardScaler

# 模擬數據
X = np.random.rand(100, 3)  # 100個樣本，3個特徵

# 使用 StandardScaler 進行標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("標準化後的數據:", X_scaled[:5])
