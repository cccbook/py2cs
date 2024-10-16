from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 加載 Iris 數據集
iris = load_iris()
X, y = iris.data, iris.target

# 將數據集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化數據
scaler = StandardScaler()

# 設定受限波茲曼機
rbm = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=25, random_state=42)

# 使用邏輯回歸作為分類器
logistic = LogisticRegression(max_iter=1000, random_state=42)

# 創建管道，將標準化、RBM 和邏輯回歸結合
classifier = Pipeline(steps=[('scaler', scaler), ('rbm', rbm), ('logistic', logistic)])

# 訓練模型
classifier.fit(X_train, y_train)

# 預測
y_pred = classifier.predict(X_test)

# 顯示分類結果報告
print(classification_report(y_test, y_pred))
