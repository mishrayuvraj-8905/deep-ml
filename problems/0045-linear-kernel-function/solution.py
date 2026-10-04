import numpy as np

def kernel_function(x1, x2):
	# Your code here
	return np.matmul(x1.T, x2)
