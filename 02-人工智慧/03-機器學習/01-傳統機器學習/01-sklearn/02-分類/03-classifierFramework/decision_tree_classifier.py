from sklearn import tree

def learn_classifier(x_train, y_train):
    classifier=tree.DecisionTreeClassifier(min_samples_split=10) # min_samples_split=150
    classifier.fit(x_train,y_train)
    return classifier
