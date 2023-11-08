class Parameter:
	from . import tensor
	def __init__(self, t, name=None):
		if not isinstance(t, tensor.tensor):
			t = tensor.tensor(t)
		self.t = t
		self.name = name