from loss import mse_loss

class Opt_regression_model:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def loss(self, p):
		def predict(p, xt):
			return p[0]+p[1]*xt
		return mse_loss(p, predict, self.x, self.y)
