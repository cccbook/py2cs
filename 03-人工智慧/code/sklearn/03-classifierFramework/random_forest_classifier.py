# OneVsRestClassifier

from sklearn.ensemble import RandomForestClassifier

def learn_classifier(x_train, y_train):
    model = RandomForestClassifier(n_estimators=25)
    model.fit(x_train, y_train)
    return model
