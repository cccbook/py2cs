import ml

# x_train,x_test,y_train,y_test=ml.load_csv('../csv/cmc.csv', 'ContraceptiveMethodUsed')
x_train,x_test,y_train,y_test=ml.load_csv('../csv/iris.csv', 'target')
# from decision_tree_classifier import train_classifier
# from mlp_classifier import learn_classifier
# from sgd_classifier import learn_classifier
# from gnb_classifier import learn_classifier
from knn_classifier import learn_classifier
# from svm_classifier import learn_classifier
# from ovr_classifier import learn_classifier
# from random_forest_classifier import learn_classifier
classifier = learn_classifier(x_train, y_train)
print('=========== train report ==========')
ml.report(classifier, x_train, y_train)
print('=========== test report ==========')
ml.report(classifier, x_test, y_test)
