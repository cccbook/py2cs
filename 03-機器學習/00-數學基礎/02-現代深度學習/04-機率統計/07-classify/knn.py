# 來源 -- https://ithelp.ithome.com.tw/articles/10197110
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = datasets.load_iris()

iris_data = iris.data
iris_label = iris.target

train_data , test_data , train_label , test_label = train_test_split(iris_data,iris_label,test_size=0.2)

knn = KNeighborsClassifier() # n_neighbors=5
# knn = KNeighborsClassifier(n_neighbors=1)
# knn = KNeighborsClassifier(n_neighbors=3)
# knn = KNeighborsClassifier(n_neighbors=37)

knn.fit(train_data,train_label)

print('預測答案：', knn.predict(test_data))
print('正確答案：', test_label)
