from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# 定義基學習器
log_clf = LogisticRegression()
dt_clf = DecisionTreeClassifier()
svm_clf = SVC(probability=True)

# 定義投票分類器，使用軟投票
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('dt', dt_clf), ('svc', svm_clf)], voting='soft')

# 訓練投票分類器
voting_clf.fit(X_train, y_train)

# 預測
y_pred = voting_clf.predict(X_test)
