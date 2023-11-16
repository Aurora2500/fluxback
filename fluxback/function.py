from .tensor import Tensor

class Function:
	def __call__(self, *args):
		targs = [arg.values if isinstance(arg, Tensor) else arg for arg in args]
		result = self.forward(*targs)
		if any(t.requires_grad for t in args):
			result = Tensor(result)
			result.requires_grad = True
			result.dependencies = [t for t in args]
			result.grad_fn = self.backward
			result.autodiff_role = self.name
		return result