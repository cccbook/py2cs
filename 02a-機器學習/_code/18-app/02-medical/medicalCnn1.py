from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 假設 X 和 y 已經定義
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 訓練 SVM 模型
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# 預測
y_pred = svm_model.predict(X_test)

# 評估
print(classification_report(y_test, y_pred))
