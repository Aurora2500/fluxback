from .function import Function
import numpy as np

class Sigmoid(Function):
	name = "Sigmoid"
	@staticmethod
	def forward(x):
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def backward(n):
		return (n.grad * n.values * (1 - n.values),)

class ReLU(Function):
	name = "ReLU"
	@staticmethod
	def forward(x):
		return np.maximum(0, x)

	@staticmethod
	def backward(n):
		return (n.grad * (n.values >= 0),)

class LeakyReLU(Function):
	name = "Leaky ReLU"
	def __init__(self, alpha=0.01):
		self.alpha = alpha

	def forward(self, x):
		return np.maximum(self.alpha * x, x)

	def backward(self, n):
		return (n.grad * (n.values >= 0) + n.grad * self.alpha * (n.values < 0),)

class SoftMax(Function):
	name="SoftMax"
	@staticmethod
	def forward(x):
		exp = np.exp(x)
		return exp / exp.sum(axis=-1, keepdims=True)
	
	@staticmethod
	def backward(n):
		size = n.values.shape[-1]
		rowwise = np.expand_dims(n.values, -1)
		colwise = np.expand_dims(n.values, -2)
		jacobian = - rowwise * colwise
		ii, jj = np.diag_indices(size, 2)
		jacobian[..., ii, jj] = n.values * (1 - n.values)
		return ((jacobian @ n.grad[..., None])[...,0],)