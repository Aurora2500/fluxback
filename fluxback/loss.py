def crossentropy(expected, predicted):
	return -np.sum(expected * np.log(predicted) + (1 - expected) * np.log(1 - predicted))