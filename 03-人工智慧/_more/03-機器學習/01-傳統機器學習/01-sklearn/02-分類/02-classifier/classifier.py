#是否避孕決策樹
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import tree
import pydot
from six import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split
data= pd.read_csv('cmc.csv')
x = data.drop(labels=['ContraceptiveMethodUsed'],axis=1).values 
y = data['ContraceptiveMethodUsed'] #是否避孕，1=No-use, 2=Long-term, 3=Short-term

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2) #切分訓練測試集

"""模型架構及訓練"""
classifier=tree.DecisionTreeClassifier(min_samples_split=150)
classifier.fit(x_train,y_train)
"""呈現"""
tree.export_graphviz(classifier,out_file='tree.dot')#圖片轉dot

dot_data = StringIO()
tree.export_graphviz(classifier, out_file=dot_data) # dot 轉圖片
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_png('tree.png')

"""預測"""
y_predicted =  classifier.predict(x_test)
print(confusion_matrix(y_test, y_predicted))
print(classification_report(y_test, y_predicted))
