from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# 加載數據集
data = load_iris()
X = data.data
y = data.target

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 創建隨機森林分類器
model = RandomForestClassifier()

# 訓練模型
model.fit(X_train, y_train)

# 預測結果
y_pred = model.predict(X_test)

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)

# 顯示混淆矩陣
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 顯示分類報告（精確率、召回率、F1 分數等）
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 如果有多類的ROC，則需要使用一對多的方式來計算AUC
# 例如，在此示例中，使用 OneVsRestClassifier 或手動分別計算每個類的 AUC
