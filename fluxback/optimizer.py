import numpy as np

class Optimizer:
	def __init__(self, params):
		self.params = params


class SGD(Optimizer):
	def __init__(self, params, lr=0.01):
		super().__init__(params)
		self.lr = lr

	def step(self):
		for param in self.params:
			param.t -= self.lr * param.t.grad