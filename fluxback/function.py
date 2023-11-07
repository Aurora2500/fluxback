from typing import Callable
from dataclasses import dataclass
from .tensor import tensor

@dataclass
class function:
	forward: Callable[..., tensor]
	backward: Callable[[tensor], tuple[float]]

	def __call__(self, *args):
		result = self.forward(*(arg.values if isinstance(arg, tensor) else arg for arg in args))
		if any(t.requires_grad for t in args):
			result = tensor(result)
			result.requires_grad = True
			result.dependencies = [t for t in args]
			result.grad_fn = lambda n: (n.grad * self.backward(n),)
		return result