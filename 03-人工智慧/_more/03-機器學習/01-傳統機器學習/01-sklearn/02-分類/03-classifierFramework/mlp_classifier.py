import tensorflow as tf
import numpy as np

class MlpClassifier:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        predict = self.model.predict(x)
        y_predicted = np.argmax(predict,axis=1)
        return y_predicted

def learn_classifier(x_train, y_train):
    """模型"""
    dim = len(x_train[0]) # 特徵數量
    category = len(np.unique(y_train)) # 答案數量
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu,input_shape=(dim,)))
    model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=category,activation=tf.nn.softmax))
    """編譯"""
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    """訓練"""
    model.fit(x_train,y_train,epochs=100)
    return MlpClassifier(model)
