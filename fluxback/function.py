from typing import Callable
from dataclasses import dataclass
from .tensor import tensor

@dataclass
class function:
	forward
	backward

	def __call__(self, *args):
		result = self.forward(*args)
		if not isinstance(result, tensor):
			return result
		result.dependencies = [t for t in args]
		result.grad_fn = self.backward
		return result