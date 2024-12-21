from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# 定義基學習器
estimators = [
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC(probability=True))
]

# 定義元學習器
meta_model = LogisticRegression()

# 定義 Stacking 模型
stacking_model = StackingClassifier(estimators=estimators, final_estimator=meta_model)

# 訓練模型
stacking_model.fit(X_train, y_train)

# 預測
y_pred = stacking_model.predict(X_test)
