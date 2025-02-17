from sklearn import svm

def learn_classifier(x_train, y_train):
    model = svm.SVC(decision_function_shape='ovo')
    model.fit(x_train, y_train)
    return model
