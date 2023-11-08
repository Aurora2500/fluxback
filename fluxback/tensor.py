from math import prod
import numpy as np
from .topology import topology_sort

class tensor:
	def __init__(self, values, requires_grad=False):
		self.values = np.array(values)
		self.requires_grad = requires_grad
		self.grad_fn = None
		self.dependencies = None
		self.zero()

	def __repr__(self):
		str_repr = str(self.values)
		str_repr = str_repr.replace("\n", "\n" + " " * len("tensor("))
		return f"tensor({str_repr})"

	def back(self):
		sorted_list = topology_sort(self)
		# make sure that self is a scalar
		if self.values.size != 1:
			raise Exception("Can only backpropagate scalar values")
		self.grad = np.ones_like(self.values)
		for node in sorted_list:
			if node.grad_fn is None:
				continue
			back_grad = node.grad_fn(node)
			for gradient, dependant in zip(back_grad, node.dependencies):
				if dependant.requires_grad:
					# edge case where the dependant is a scalar and the gradient is a vector
					if dependant.grad.shape != gradient.shape:
						gradient = gradient.sum()
					dependant.grad += gradient

	def zero(self):
		self.grad = np.zeros_like(self.values)
		if self.dependencies is not None:
			for dep in self.dependencies:
				dep.zero()

	def __pos__(self):
		return self

	def __neg__(self):
		result = tensor(-self.values)
		if self.requires_grad:
			result.requires_grad = True
			result.dependencies = [self]
			result.grad_fn = lambda node: (-node.grad,)
		return result

	def __abs__(self):
		result = tensor(abs(self.values))
		if self.requires_grad:
			result.requires_grad = True
			result.dependencies = [self]
			result.grad_fn = lambda node: (node.grad * np.sign(node.dependencies[0].values),)
		return result

	def __add__(self, other):
		if not isinstance(other, tensor):
			other = tensor(other)
		result = tensor(self.values + other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad, n.grad)
		return result

	def __radd__(self, other):
		return self + other

	def __sub__(self, other):
		if not isinstance(other, tensor):
			other = tensor(other)
		result = tensor(self.values - other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad, -n.grad)
		return result

	def __rsub__(self, other):
		if not isinstance(other, tensor):
			other = tensor(other)
		return other - self

	def __mul__(self, other):
		if not isinstance(other, tensor):
			other = tensor(other)
		result = tensor(self.values * other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad * n.dependencies[1].values, n.grad * n.dependencies[0].values)
		return result

	def __rmul__(self, other):
		return self * other

	def __truediv__(self, other):
		if not isinstance(other, tensor):
			other = tensor(other)
		result = tensor(self.values / other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad / n.dependencies[1].values, -n.grad * n.dependencies[0].values / n.dependencies[1].values ** 2)
		return result

	def __rtruediv__(self, other):
		if not isinstance(other, tensor):
			other = tensor(other)
		return other / self
	
	def __pow__(self, other):
		if not isinstance(other, tensor):
			other = tensor(other)
		result = tensor(self.values ** other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad * n.dependencies[1].values * n.dependencies[0].values ** (n.dependencies[1].values - 1),
																	n.grad * np.log(n.dependencies[0].values) * n.dependencies[0].values ** n.dependencies[1].values)
		return result

	def __rpow__(self, other):
		if not isinstance(other, tensor):
			other = tensor(other)
		return other ** self

	def __matmul__(self, other):
		if not isinstance(other, tensor):
			other = tensor(other)
		result = tensor(self.values @ other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad @ n.dependencies[1].values.T, n.dependencies[0].values.T @ n.grad)
		return result

	def __rmatmul__(self, other):
		if not isinstance(other, tensor):
			other = tensor(other)
		return other @ self

	def exp(self):
		result = tensor(np.exp(self.values))
		if self.requires_grad:
			result.requires_grad = True
			result.dependencies = [self]
			result.grad_fn = lambda n: (n.grad * np.exp(n.dependencies[0].values),)
		return result

	def sum(self):
		result = tensor(self.values.sum())
		if self.requires_grad:
			result.requires_grad = True
			result.dependencies = [self]
			result.grad_fn = lambda n: (n.grad * np.ones_like(n.dependencies[0].values),)
		return result
	
	def max(self, other):
		result = tensor(np.maximum(self.values, other.values))
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad * (n.dependencies[0].values >= n.dependencies[1].values),
																	n.grad * (n.dependencies[0].values < n.dependencies[1].values))
		return result

	def reshape(self, new_shape):
		result = tensor(self.values.reshape(new_shape))
		if self.requires_grad:
			result.requires_grad = True
			result.dependencies = [self]
			result.grad_fn = lambda n: (n.grad.reshape(n.dependencies[0].values.shape),)
		return result

	@staticmethod
	def zeros(shape, requires_grad=False):
		return tensor(np.zeros(shape), requires_grad=requires_grad)

	@staticmethod
	def ones(shape, requires_grad=False):
		return tensor(np.ones(shape), requires_grad=requires_grad)

	@staticmethod
	def randn(shape, requires_grad=False):
		return tensor(np.random.randn(*shape), requires_grad=requires_grad)
