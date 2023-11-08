from .tensor import tensor

class Function:
	def __call__(self, *args):
		result = self.forward(*(arg.values if isinstance(arg, tensor) else arg for arg in args))
		if any(t.requires_grad for t in args):
			result = tensor(result)
			result.requires_grad = True
			result.dependencies = [t for t in args]
			result.grad_fn = self.backward
		return result