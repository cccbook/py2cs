from sklearn import datasets

iris = datasets.load_iris()
# print(iris)
print('iris.keys()=', iris.keys())
print('filename=', iris.filename) # 特徵屬性的名稱
print('feature_names=', iris.feature_names) # 特徵屬性的名稱
print('data=', iris.data) # data 是一個 numpy 2d array, 通常用 X 代表
print('target=', iris.target) # target 目標值，答案，通常用 y 代表
print('target_names=', iris.target_names) # 目標屬性的名稱
print('DESCR=', iris.DESCR)

import pandas as pd

x = pd.DataFrame(iris.data, columns=iris.feature_names)
print('x=', x)
y = pd.DataFrame(iris.target, columns=['target'])
print('y=', y)
data = pd.concat([x,y], axis=1) # 水平合併 x | y
# axis 若為 0 代表垂直合併  x/y (x over y)
print('data=\n', data)
data.to_csv('iris_dump.csv', index=False)
