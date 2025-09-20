import numpy as np


def addNoise(data, percent):
	np.random.seed(0)
	std_data = np.std(data, axis=0)
	std_data = np.reshape(std_data, (1, -1))
	std_data = np.repeat(std_data, np.shape(data)[0], axis=0)
	noise = np.multiply(np.random.randn(*np.shape(data)), percent/1000.0*std_data)
	data += noise
	return data