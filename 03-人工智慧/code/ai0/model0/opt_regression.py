class Opt_regression_model:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def predict(self, p, xt):
		return p[0]+p[1]*xt

	def MSE(self, p):
		total = 0
		for i in range(len(self.x)):
			total += (self.y[i]-self.predict(p,self.x[i]))**2
		return total

	def loss(self, p):
		return self.MSE(p)
