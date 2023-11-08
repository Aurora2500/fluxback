def crossentropy(expected, predicted):
	return -np.sum(expected * np.log(predicted) + (1 - expected) * np.log(1 - predicted))

def mse(expected, predicted):
	return np.sum((expected - predicted) ** 2) / len(expected)

def mae(expected, predicted):
	return np.sum(np.abs(expected - predicted)) / len(expected)