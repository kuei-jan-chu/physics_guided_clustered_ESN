import numpy as np
from scipy import ndimage, stats


def transfer_entropy(X,Y,delay=1,gaussian_sigma=None):
	'''
	TE implementation: asymmetric statistic measuring the reduction in uncertainty
	for a future value of X given the history of X and Y.
	Calculated through the Kullback-Leibler divergence with conditional probabilities

	Quantifies the amount of information from Y to X.

	args:
		- X (1D array):
			time series of scalars (1D array)
		- Y (1D array):
			time series of scalars (1D array)
	kwargs:
		- delay (int): 
			step in tuple (x_n, y_{n - delay}, x_(n - delay))
		- gaussian_sigma (int):
			sigma to be used
			default set at None: no gaussian filtering applied
	returns:
		- TE (float):
			transfer entropy between X and Y given the history of X
	'''

	if len(X)!=len(Y):
		raise ValueError('time series entries need to have same length')

	n = float(len(X[delay:]))

	# number of bins for X and Y using Freeman-Diaconis rule
	# histograms built with numpy.histogramdd
	binX = int( (max(X)-min(X))
				/ (2* stats.iqr(X) / (len(X)**(1.0/3))) )
	binY = int( (max(Y)-min(Y))
				/ (2* stats.iqr(Y) / (len(Y)**(1.0/3))) )

	# Definition of arrays of shape (D,N) to be transposed in histogramdd()
	data_joint_Xfut_Xpast_Ypast = np.array([X[delay:],X[:-delay],Y[:-delay]])
	data_joint_Xpast_Ypast = np.array([X[:-delay],Y[:-delay]])
	data_joint_Xfut_Xpast = np.array([X[delay:],X[:-delay]])
	data_Xpast = np.array(X[:-delay])

	# p_Xfut_Xpast_Ypast : a 3D array of counts:
	#   p_Xfut_Xpast_Ypast[i, j, k] is the number of samples falling into the i-th bin along X’s future,
	#   the j-th bin along Y’s past, the k-th bin along X’s past.
	# bin_p3 : a list of 3 arrays, each array giving the bin edges along one of the three dimensions.
	p_Xfut_Xpast_Ypast,bin_edges_Xfut_Xpast_Ypast = np.histogramdd(
		sample = data_joint_Xfut_Xpast_Ypast.T,
		bins = [binX,binX,binY])

	p_Xpast_Ypast,bin_edges_Xpast_Ypast = np.histogramdd(
		sample = data_joint_Xpast_Ypast.T,
		bins=[binX,binY])

	p_Xfut_Xpast,bin_edges_Xfut_Xpast = np.histogramdd(
		sample = data_joint_Xfut_Xpast.T,
		bins=[binX,binX])

	p_Xpast,bin_edges_Xpast = np.histogramdd(
		sample = data_Xpast,
		bins=binX)

	# Hists normalized to obtain densities
	p_Xpast = p_Xpast/n
	p_Xpast_Ypast = p_Xpast_Ypast/n
	p_Xfut_Xpast = p_Xfut_Xpast/n
	p_Xfut_Xpast_Ypast = p_Xfut_Xpast_Ypast/n   

	if gaussian_sigma is not None:
		s = gaussian_sigma
		p_Xpast = ndimage.gaussian_filter(p_Xpast, sigma=s)
		p_Xpast_Ypast = ndimage.gaussian_filter(p_Xpast_Ypast, sigma=s)
		p_Xfut_Xpast = ndimage.gaussian_filter(p_Xfut_Xpast, sigma=s)
		p_Xfut_Xpast_Ypast = ndimage.gaussian_filter(p_Xfut_Xpast_Ypast, sigma=s)

	# Calculating elements in TE summation
	elements = []
	for i_xfut in range(p_Xfut_Xpast_Ypast.shape[0]):
		for i_xpast in range(p_Xfut_Xpast_Ypast.shape[1]):
			for i_ypast in range(p_Xfut_Xpast_Ypast.shape[2]):
				p_Xpast_val = p_Xpast[i_xpast]
				p_Xpast_Ypast_val = p_Xpast_Ypast[i_xpast][i_ypast]
				p_Xfut_Xpast_val = p_Xfut_Xpast[i_xfut][i_xpast]
				p_Xfut_Xpast_Ypast_val = p_Xfut_Xpast_Ypast[i_xfut][i_xpast][i_ypast]

				arg1 = float(p_Xpast_Ypast_val * p_Xfut_Xpast_val)
				arg2 = float(p_Xfut_Xpast_Ypast_val * p_Xpast_val)

				# Corrections avoding log(0)
				if arg1 == 0.0: arg1 = float(1e-8)
				if arg2 == 0.0: arg2 = float(1e-8)

				term = p_Xfut_Xpast_Ypast_val* (np.log2(arg2) - np.log2(arg1))
				elements.append(term)
	TE = np.sum(elements)

	return TE

def transfer_entropy_matrix(data_matrix, delay=1, gaussian_sigma=None):
	_, dim = data_matrix.shape

	# Initialize matrix
	TE_matrix = np.zeros((dim, dim))

	# Compute pairwise TE
	for i in range(dim):
		for j in range(dim):
			if i == j:
				continue  # skip self-TE
				# X = data_matrix[:, i]
				# Y = data_matrix[:, j]
				# TE_matrix[i, j] = transfer_entropy(X, Y, delay, gaussian_sigma=gaussian_sigma)
				# print(f"TE({i}, {j}) = {te}")
			X = data_matrix[:, i]
			Y = data_matrix[:, j]
			TE_matrix[i, j] = transfer_entropy(X, Y, delay, gaussian_sigma=gaussian_sigma)

	return TE_matrix