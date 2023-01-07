# https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes
from sklearn.naive_bayes import GaussianNB

def learn_classifier(x_train, y_train):
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model
