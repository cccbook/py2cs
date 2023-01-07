from sklearn.linear_model import SGDClassifier

def learn_classifier(x_train, y_train):
    model = SGDClassifier(loss="log_loss", penalty="l2", max_iter=100)
    model.fit(x_train, y_train)
    return model
