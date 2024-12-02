#是否避孕MLP
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
"""資料準備"""
data= pd.read_csv('cmc.csv')
x = data.drop(labels=['ContraceptiveMethodUsed'],axis=1).values 
y = data['ContraceptiveMethodUsed']-1 #是否避孕，1=No-use, 2=Long-term, 3=Short-term
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2) #切分訓練測試集

"""模型"""
dim = 9 #九個特徵
category = 3 #三種答案
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu,input_shape=(dim,)))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=category,activation=tf.nn.softmax))
"""編譯"""
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
"""訓練"""
model.fit(x_train,y_train,epochs=100)

"""預測"""
predict= model.predict(x_test)
print(f'predict={predict}')
y_predicted = np.argmax(predict,axis=1) + 1
y_test = y_test+1
print(confusion_matrix(y_test, y_predicted))
print(classification_report(y_test, y_predicted))