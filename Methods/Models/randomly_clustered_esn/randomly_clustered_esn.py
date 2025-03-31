#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by:  Jaideep Pathak, University of Maryland
                Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import sys

import numpy as np
import torch
from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from plotting_utils import *
from global_utils import *
from esn import ESN

class RandClusteredESN(ESN):
        
    def __init__(self, params):
        super().__init__(params)

        # Parameters specific to PGClusteredESN
        self.input_group_size = params["input_group_size"]
        self.num_clusters = int(self.input_dim / self.input_group_size)

        self.nodes_per_group = int(np.ceil(self.reservoir_size/self.num_clusters))
        # self.reservoir_size = int(self.num_clusters * self.nodes_per_group)

        self.cluster_size = self.nodes_per_group


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
        'input_group_size': 'IGS',
        'worker_id':'WID', 
        }
        return keys

    def getReservoirWeights(self, size_x, size_y, radius, sparsity):
        # Setting the reservoir size automatically to avoid overfitting
        if self.display_output == True : print("Initializing the reservoir weights...")
        if self.display_output == True : print("NETWORK SPARSITY: {:}".format(self.sparsity))
        if self.display_output == True : print("Computing sparse hidden to hidden weight matrix...")

        in_cluster_percentage = self.input_group_size / self.input_dim
        between_clusters_sparsity = (sparsity - in_cluster_percentage) / (1 - in_cluster_percentage)
        cluster_size = self.cluster_size

        W_h = torch.rand((size_x, size_y))
        mask = torch.zeros((size_x, size_y), dtype=bool)
        sparsity_mask = torch.ones((size_x, size_y),  dtype=bool)

        for i in range(self.num_clusters):
            start_index = i * cluster_size
            end_index = (i+1) * cluster_size
            
            # Connect within each cluster
            mask[start_index:end_index, start_index:end_index] = True  # Mask the cluster block
        
        sparsity_mask[~mask] = (torch.rand(size_x, size_y)[~mask] < between_clusters_sparsity)

        W_h = W_h * sparsity_mask
        # Scale W to achieve the specified spectral radius
        if self.display_output: print("EIGENVALUE DECOMPOSITION")
        eigenvalues = torch.linalg.eigvals(W_h).abs()
        W_h = (W_h / eigenvalues.max()) * radius  # Scale to the spectral radius

        return W_h

    # def getInputLayerWeights(self, reservoir_size, input_dim, sigma_input, sparsity):
    #     if self.display_output == True : print("Initializing the input weights...")
    #     # Initialize input layer weights with sparsity
    #     W_in = torch.rand(reservoir_size, input_dim) * 2 * sigma_input - sigma_input  # Values between -sigma_input and sigma_input
    #     sparsity_mask = (torch.rand(reservoir_size, input_dim) < sparsity)
    #     W_in = W_in * sparsity_mask  # Apply sparsity

    #     return W_in

    def getInputLayerWeights(self, reservoir_size, input_dim, sigma_input, sparsity):
        if self.display_output == True : print("Initializing the input weights...")
        num_clusters = self.num_clusters
        cluster_size = self.cluster_size
        input_group_size = self.input_group_size

        in_cluster_percentage = self.input_group_size / self.input_dim
        outside_clusters_sparsity = (sparsity - in_cluster_percentage) / (1 - in_cluster_percentage)

        # Initialize input layer weights with sparsity
        W_in = (torch.rand(reservoir_size, input_dim) * 2 - 1) * sigma_input  # Values between -sigma_input and sigma_input
        mask = torch.zeros((reservoir_size, input_dim), dtype=bool)
        sparsity_mask = torch.ones((reservoir_size, input_dim),  dtype=bool)

        for i in range(num_clusters):
            start_index = i * cluster_size
            end_index = (i+1) * cluster_size

            # connect with its corresponding input dimensions
            corresponded_start_input_dim = i * input_group_size
            corresponded_end_input_dim = (i+1) * input_group_size

            mask[start_index:end_index, corresponded_start_input_dim:corresponded_end_input_dim] = True  # Mask the cluster block

        sparsity_mask[~mask] = (torch.rand(reservoir_size, input_dim)[~mask] < outside_clusters_sparsity)
        W_in = W_in * sparsity_mask  # Apply sparsity

        return W_in