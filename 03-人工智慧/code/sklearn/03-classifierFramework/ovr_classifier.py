# OneVsRestClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def learn_classifier(x_train, y_train):
    model = OneVsRestClassifier(LinearSVC(random_state=0))
    model.fit(x_train, y_train)
    return model
