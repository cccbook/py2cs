from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# 定義基學習器
base_learners = [
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC())
]

# 定義元學習器
meta_learner = LogisticRegression()

# 定義 Stacking 模型
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)

# 訓練 Stacking 模型
stacking_clf.fit(X_train, y_train)

# 預測
y_pred = stacking_clf.predict(X_test)
