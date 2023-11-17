import numpy as np
from .topology import topology_sort

from graphviz import Digraph

class Tensor:
	def __init__(self, values, requires_grad=False):
		self.autodiff_role = "Const"
		if isinstance(values, ComputationalTensor):
			self.values = values.values
			self.requires_grad = values.requires_grad or requires_grad
			return
		self.values = np.array(values)
		self.requires_grad = requires_grad

	def __repr__(self):
		str_repr = str(self.values)
		str_repr = str_repr.replace("\n", "\n" + " " * len("tensor("))
		return f"tensor({str_repr})"

	def __pos__(self):
		return self

	def __neg__(self):
		result = ComputationalTensor(-self.values, requires_grad=self.requires_grad, dependencies=[self])
		if self.requires_grad:
			result.grad_fn = lambda node: (-node.grad,)
			result.autodiff_role = "Neg"
		return result

	def __abs__(self):
		result = ComputationalTensor(abs(self.values), requires_grad=self.requires_grad, dependencies=[self])
		if self.requires_grad:
			result.requires_grad = True
			result.dependencies = [self]
			result.grad_fn = lambda node: (node.grad * np.sign(node.dependencies[0].values),)
			result.autodiff_role = "Abs"
		return result

	def __add__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		result = ComputationalTensor(self.values + other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad, n.grad)
			result.autodiff_role = "Add"
		return result

	def __radd__(self, other):
		return self + other

	def __iadd__(self, other):
		if isinstance(other, Tensor):
			self.values += other.values
		if isinstance(other, np.ndarray):
			self.values += other
		return self

	def __sub__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		result = ComputationalTensor(self.values - other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad, -n.grad)
			result.autodiff_role = "Sub"
		return result

	def __rsub__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		return other - self

	def __isub__(self, other):
		if isinstance(other, Tensor):
			self.values -= other.values
		else:
			self.values -= other
		return self

	def __mul__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		result = ComputationalTensor(self.values * other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad * n.dependencies[1].values, n.grad * n.dependencies[0].values)
			result.autodiff_role = "Mul"
		return result

	def __rmul__(self, other):
		return self * other

	def __imul__(self, other):
		if isinstance(other, Tensor):
			self.values *= other.values
		if isinstance(other, np.ndarray):
			self.values *= other
		return self

	def __truediv__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		result = ComputationalTensor(self.values / other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad / n.dependencies[1].values, -n.grad * n.dependencies[0].values / n.dependencies[1].values ** 2)
			result.autodiff_role = "Div"
		return result

	def __rtruediv__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		return other / self

	def __itruediv__(self, other):
		if isinstance(other, Tensor):
			self.values /= other.values
		if isinstance(other, np.ndarray):
			self.values /= other
		return self

	def __pow__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		result = ComputationalTensor(self.values ** other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad * n.dependencies[1].values * n.dependencies[0].values ** (n.dependencies[1].values - 1),
																	n.grad * np.log(n.dependencies[0].values) * n.dependencies[0].values ** n.dependencies[1].values)
			result.autodiff_role = "Pow"
		return result

	def __rpow__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		return other ** self

	def __ipow__(self, other):
		if isinstance(other, Tensor):
			self.values **= other.values
		if isinstance(other, np.ndarray):
			self.values **= other
		return self

	def __matmul__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		result = ComputationalTensor(self.values @ other.values)
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad @ n.dependencies[1].values.T, n.dependencies[0].values.T @ n.grad)
			result.autodiff_role = "Matmul"
		return result

	def __rmatmul__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		return other @ self

	def __imatmul__(self, other):
		if isinstance(other, Tensor):
			self.values **= other.values
		if isinstance(other, np.ndarray):
			self.values @= other
		return self

	def exp(self):
		result = ComputationalTensor(np.exp(self.values))
		if self.requires_grad:
			result.requires_grad = True
			result.dependencies = [self]
			result.grad_fn = lambda n: (n.grad * np.exp(n.dependencies[0].values),)
			result.autodiff_role = "Exp"
		return result

	def log(self):
		result = ComputationalTensor(np.log(self.values))
		if self.requires_grad:
			result.requires_grad = True
			result.dependencies = [self]
			result.grad_fn = lambda n: (n.grad / n.dependencies[0].values,)
			result.autodiff_role = "Log"
		return result

	def sum(self):
		result = ComputationalTensor(self.values.sum())
		if self.requires_grad:
			result.requires_grad = True
			result.dependencies = [self]
			result.grad_fn = lambda n: (n.grad * np.ones_like(n.dependencies[0].values),)
			result.autodiff_role = "Sum"
		return result
	
	def max(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		result = ComputationalTensor(np.maximum(self.values, other.values))
		if self.requires_grad or other.requires_grad:
			result.requires_grad = True
			result.dependencies = [self, other]
			result.grad_fn = lambda n: (n.grad * (n.dependencies[0].values >= n.dependencies[1].values),
																	n.grad * (n.dependencies[0].values < n.dependencies[1].values))
			result.autodiff_role = "Max"
		return result

	def reshape(self, new_shape):
		result = ComputationalTensor(self.values.reshape(new_shape))
		if self.requires_grad:
			result.requires_grad = True
			result.dependencies = [self]
			result.grad_fn = lambda n: (n.grad.reshape(n.dependencies[0].values.shape),)
			result.autodiff_role = "Reshape"
		return result

	@classmethod
	def zeros(cls, shape, requires_grad=False):
		return cls(np.zeros(shape), requires_grad=requires_grad)

	@classmethod
	def ones(cls, shape, requires_grad=False):
		return cls(np.ones(shape), requires_grad=requires_grad)

	@classmethod
	def randn(cls, shape, requires_grad=False):
		return cls(np.random.randn(*shape), requires_grad=requires_grad)

	@property
	def dependencies(self):
		return None

class ComputationalTensor(Tensor):

	def __init__(self, values, requires_grad=False, dependencies=None):
		if requires_grad and dependencies is None:
			raise Exception("Cannot require grad without dependencies")
		super().__init__(values, requires_grad=requires_grad)
		self._dependencies = dependencies if dependencies is not None else []

	@property
	def dependencies(self):
		return self._dependencies

	@dependencies.setter
	def dependencies(self, value):
		self._dependencies = value

	def __iadd__(self, other):
		raise Exception("Inplace operations are not supported")

	def __isub__(self, other):
		raise Exception("Inplace operations are not supported")

	def __imul__(self, other):
		raise Exception("Inplace operations are not supported")

	def __itruediv__(self, other):
		raise Exception("Inplace operations are not supported")

	def __ipow__(self, other):
		raise Exception("Inplace operations are not supported")

	def __imatmul__(self, other):
		raise Exception("Inplace operations are not supported")

	def back(self):
		sorted_list = topology_sort(self)
		# make sure that self is a scalar
		if self.values.size != 1:
			raise Exception("Can only backpropagate scalar values")
		self.grad = np.ones_like(self.values)
		for node in sorted_list:
			if not isinstance(node, ComputationalTensor) or node.grad_fn is None:
				continue
			back_grad = node.grad_fn(node)
			for gradient, dependant in zip(back_grad, node.dependencies):
				if dependant.requires_grad:
					# edge case where the dependant is a scalar and the gradient is a vector
					if dependant.grad.size == 1 and gradient.size != 1:
						gradient = gradient.sum()
					# edge case found when dealing with bias and baches
					if len(dependant.grad.shape) != 0 and dependant.grad.shape[-1] == gradient.shape[-1] and len(dependant.grad.shape) < len(gradient.shape):
						gradient = gradient.sum(axis=tuple(range(len(gradient.shape) - 1)))
					dependant.grad += gradient

	def zero(self):
		self.grad = np.zeros_like(self.values)
		stack = [self]
		while len(stack) > 0:
			node = stack.pop()
			if node.requires_grad:
				node.grad = np.zeros_like(node.values)
				if isinstance(node, ComputationalTensor):
					for dep in node.dependencies:
							stack.append(dep)

	def graphviz(self):
		dot = Digraph()
		node_map = {}
		node_stack = [self]
		edges = []
		counter = 1
		while len(node_stack) > 0:
			node = node_stack.pop()
			if node in node_map:
				continue
			node_map[node] = counter
			dot.node(str(counter), node.autodiff_role)
			counter += 1
			if node.dependencies is None:
				continue
			for dep in node.dependencies:
				edges.append((node, dep))
				node_stack.append(dep)
		for (a, b) in edges:
			dot.edge(str(node_map[b]), str(node_map[a]))
		return dot
