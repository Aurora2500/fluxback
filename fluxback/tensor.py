from math import prod
import numpy as np
from .topology import topology_sort

class tensor:

	def __init__(self, values, requires_grad=False, grad_fn=None):
		self.values = np.array(values)
		self.requires_grad = requires_grad
		self.grad_fn = grad_fn
		self.dependencies = None
		self.zero()
	
	def __repr__(self):
		return f"tensor({self.values})"
	
	def zero(self):
		self.grad = np.zeros_like(self.values)
		if self.dependencies is not None:
			for dep in self.dependencies:
				dep.zero()
	
	def back(self):
		sorted_list = topology_sort(self)
		self.grad = np.ones_like(self.values)
		for node in sorted_list:
			if node.grad_fn is None:
				continue
			back_grad = node.grad_fn(node)
			for gradient, dependant in zip(back_grad, node.dependencies):
				dependant.grad += gradient
	
	def __add__(self, other):
		def add_grad(node):
			return tuple(node.grad for _ in node.dependencies)
		if not isinstance(other, tensor):
			other = tensor(other)
		result = tensor(self.values + other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = add_grad
		return result

	def __mul__(self, other):
		def mul_grad(node):
			return tuple(
				node.grad * prod(x.values for j, x in enumerate(node.dependencies) if i != j)
				for i in range(len(node.dependencies))
			)
		if not isinstance(other, tensor):
			other = tensor(other)
		result = tensor(self.values * other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = mul_grad
		return result
