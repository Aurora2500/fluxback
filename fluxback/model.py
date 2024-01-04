class Model:
	def __init__(self):
		self.built = False

	def build(self):
		if self.built:
			raise Exception("Model already built")
		return ModelBuilder(self)


class ModelBuilder:
	def __init__(self, model):
		self.model = model
		parameters = []

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		pass