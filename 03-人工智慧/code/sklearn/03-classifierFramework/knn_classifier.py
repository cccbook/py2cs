# https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes
from sklearn.neighbors import KNeighborsClassifier

def learn_classifier(x_train, y_train):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)
    return model
