from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# 初始化三個模型
model1 = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=500)
model2 = SVC(kernel='linear', probability=True)
model3 = LogisticRegression()

# 使用 VotingClassifier 將三個模型融合
voting_clf = VotingClassifier(estimators=[('mlp', model1), ('svm', model2), ('logreg', model3)], voting='soft')

# 訓練模型
voting_clf.fit(X_train, y_train)

# 評估模型
y_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Voting Classifier Accuracy: {accuracy:.4f}")
