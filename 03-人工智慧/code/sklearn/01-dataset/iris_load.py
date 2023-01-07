import pandas as pd

data= pd.read_csv('./iris_dump.csv')
print('data=\n', data)

x = data.drop(labels=['target'],axis=1).values # 垂直欄 target 去掉
print('x=', x)

y = data['target']
print('y=', y)
