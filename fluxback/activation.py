from .function import Function
import numpy as np

class Sigmoid(Function):
	@staticmethod
	def forward(x):
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def backward(n):
		return (n.grad * n.values * (1 - n.values),)

class ReLU(Function):
	@staticmethod
	def forward(x):
		return np.maximum(0, x)

	@staticmethod
	def backward(n):
		return (n.grad * (n.values >= 0),)

class LeakyReLU(Function):
	def __init__(self, alpha=0.01):
		self.alpha = alpha

	def forward(self, x):
		return np.maximum(self.alpha * x, x)

	def backward(self, n):
		return (n.grad * (n.values >= 0) + n.grad * self.alpha * (n.values < 0),)

class SoftMax(Function):
	@staticmethod
	def forward(x):
		exp = np.exp(x)
		return exp / exp.sum(axis=1, keepdims=True)
	
	@staticmethod
	def backward(n):
		return (n.grad * n.values * (1 - n.values),)