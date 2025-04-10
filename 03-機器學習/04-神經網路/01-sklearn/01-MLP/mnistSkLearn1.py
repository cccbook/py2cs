# https://chatgpt.com/c/670f19e5-4da8-8012-8204-d1c389a9b797

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# 加載數據集
digits = load_digits()
X, y = digits.data, digits.target

# 將數據集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化並訓練 MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=300, activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, y_train)

# 預測並生成分類報告
y_pred = mlp.predict(X_test)
print(classification_report(y_test, y_pred))
