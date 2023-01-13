import matplotlib.pyplot as plt
from sklearn import linear_model
reg = linear_model.LinearRegression()
x = [[1], [2], [3]]
y = [3.0, 5.0, 7.0] # y = [3.1, 5.0, 7.1]
reg.fit(x, y)
print('c0=', reg.predict([[0.0]]))
print('c=', reg.coef_)
y_predicted = reg.predict(x)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()
