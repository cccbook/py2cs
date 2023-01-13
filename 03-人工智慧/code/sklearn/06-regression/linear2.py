from sklearn import linear_model
reg = linear_model.LinearRegression()
x = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
reg.fit(x, y)
print(reg.coef_)
