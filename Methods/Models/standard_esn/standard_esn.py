import sys
import torch
from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from plotting_utils import *
from global_utils import *
from esn import ESN


class StandardESN(ESN):
    def __init__(self, params):
        super().__init__(params)    

    def getKeysInModelName(self):
        keys = {
        'RDIM':'RDIM', 
        'N_train':'N_train', 
        'N_test':'N_test',
        'approx_reservoir_size':'SIZE', 
        'sparsity':'SP', 
        'radius':'RADIUS',
        'sigma_input':'SIGMA',
        'p_in': 'PIN',
        'dynamics_length':'DL',
        'noise_level':'NL',
        'iterative_prediction_length':'IPL',
        'regularization':'REG',
        'worker_id':'WID', 
        }
        return keys

    def getReservoirWeights(self, size_x, size_y, radius, sparsity):
        # Setting the reservoir size automatically to avoid overfitting
        if self.display_output == True : print("Initializing the reservoir weights...")
        if self.display_output == True : print("NETWORK SPARSITY: {:}".format(self.sparsity))
        if self.display_output == True : print("Computing sparse hidden to hidden weight matrix...")

        # Initialize sparse weights with PyTorch
        # W_h = torch.rand(size_x, size_y) * 2 - 1  # Values between -1 and 1
        W_h = torch.rand(size_x, size_y)
        sparsity_mask = (torch.rand(size_x, size_y) < sparsity)  # Mask for sparsity
        W_h = W_h * sparsity_mask  # Apply sparsity

        # Scale W to achieve the specified spectral radius
        if self.display_output:
            print("EIGENVALUE DECOMPOSITION")
        eigenvalues = torch.linalg.eigvals(W_h).abs()
        W_h = (W_h / eigenvalues.max()) * radius  # Scale to the spectral radius

        return W_h

    def getInputLayerWeights(self, reservoir_size, input_dim, sigma_input, sparsity):
        if self.display_output == True : print("Initializing the input weights...")
        # Initialize input layer weights with sparsity
        W_in = torch.rand(reservoir_size, input_dim) * 2 * sigma_input - sigma_input  # Values between -sigma_input and sigma_input
        sparsity_mask = (torch.rand(reservoir_size, input_dim) < sparsity)
        W_in = W_in * sparsity_mask  # Apply sparsity

        return W_in