import numpy as np
from .function import function

def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

sigmoid = function(_sigmoid, lambda n: (_sigmoid(n) * (1. - _sigmoid(n))))
