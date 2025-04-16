from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. 載入資料集
iris = load_iris()
X = iris.data
y = iris.target

# 2. 分割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 建立模型
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# 4. 訓練模型
clf.fit(X_train, y_train)

# 5. 預測
y_pred = clf.predict(X_test)

# 6. 評估模型
acc = accuracy_score(y_test, y_pred)
print(f"準確率：{acc:.2f}")

# 7. 視覺化決策樹
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
