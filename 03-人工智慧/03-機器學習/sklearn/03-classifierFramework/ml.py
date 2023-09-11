import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split

def load_csv(csvFile, target_name):
    data= pd.read_csv(csvFile)
    x = data.drop(labels=[target_name],axis=1).values
    y0 = list(data[target_name])
    slot = list(np.unique(y0))
    y1 = [slot.index(value) for value in y0] # 將 target 映射到 0...n
    y = np.array(y1) # dtype=np.single
    return train_test_split(x,y,test_size=0.2) #切分訓練測試集

def report(classifier, x_test, y_test):
    y_predicted = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_predicted))
    print(classification_report(y_test, y_predicted))
