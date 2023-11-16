from . import Tensor

class Parameter(Tensor):
	def __init__(self, values, requires_grad=True):
		super().__init__(values, requires_grad=requires_grad)

