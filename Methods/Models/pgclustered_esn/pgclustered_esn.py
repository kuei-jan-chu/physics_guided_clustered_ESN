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


class PGClusteredESN(ESN):
    def __init__(self, params):
        super().__init__(params)    

        # Parameters specific to PGClusteredESN
        self.in_cluster_weight = params["in_cluster_weight"]
        self.coupling_dims = params["coupling_dims"]
        
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
        'in_cluster_weight': 'ICW',
        'worker_id':'WID', 
        }
        return keys

    def getReservoirWeights(self, size_x, size_y, radius, sparsity):
        if self.display_output == True : print("Initializing the reservoir weights...")
        if self.display_output == True : print("NETWORK SPARSITY: {:}".format(self.sparsity))
        if self.display_output == True : print("Computing sparse hidden to hidden weight matrix...")
        
        cluster_size = self.cluster_size
        num_clusters = self.num_clusters
        W_h = torch.zeros((size_x, size_y))

        for i in range(num_clusters):
            start_index = i * cluster_size
            end_index = (i+1) * cluster_size
            
            # Connect within each cluster
            W_h[start_index:end_index, start_index:end_index] = self.in_cluster_weight * torch.rand(cluster_size, cluster_size)
            # W_h[start_index:end_index, start_index:end_index] = torch.rand(cluster_size, cluster_size)


            # Connect to the coupled clusters
            for j in self.coupling_dims:
                coupled_cluster_start_idx = ((i + j) % num_clusters) * cluster_size
                coupled_cluster_end_idx = coupled_cluster_start_idx + cluster_size
                W_h[start_index:end_index, coupled_cluster_start_idx:coupled_cluster_end_idx] = \
                    (1 - self.in_cluster_weight) * torch.rand(cluster_size, cluster_size)
                # W_h[start_index:end_index, coupled_cluster_start_idx:coupled_cluster_end_idx] = torch.rand(cluster_size, cluster_size)
                
        # no need of sparsity_mask as we are using coupling_dims to determine the connections
        # sparsity_mask = (torch.rand(size_x, size_y) < sparsity)  # Mask for sparsity
        # W_h = W_h * sparsity_mask

        # to print the values do W.A
        if self.display_output: print("EIGENVALUE DECOMPOSITION")
        eigenvalues = torch.linalg.eigvals(W_h).abs()
        W_h = (W_h / eigenvalues.max()) * radius  # Scale to the spectral radius
        
        return W_h


    def getInputLayerWeights(self, reservoir_size, input_dim, sigma_input, sparsity):
        if self.display_output == True : print("Initializing the input weights...")
        W_in = torch.zeros((reservoir_size, input_dim))

        num_clusters = self.num_clusters
        cluster_size = self.cluster_size
        input_group_size = self.input_group_size
        for i in range(num_clusters):
            start_index = i * cluster_size
            end_index = (i+1) * cluster_size

            # connect with its corresponding input dimensions
            corresponded_start_input_dim = i * input_group_size
            corresponded_end_input_dim = (i+1) * input_group_size
            W_in[start_index:end_index, corresponded_start_input_dim:corresponded_end_input_dim] = (torch.rand(cluster_size, input_group_size)*2 - 1) * sigma_input
            
            # Connect to the coupled input dimensions    
            for j in self.coupling_dims:
                coupled_start_input_dim = ((i + j) % num_clusters) * input_group_size
                coupled_end_input_dim = coupled_start_input_dim + input_group_size
                W_in[start_index:end_index, coupled_start_input_dim:coupled_end_input_dim] = (torch.rand(cluster_size, input_group_size)*2 - 1) * sigma_input
        
        # no need of sparsity_mask as we are using coupling_dims to determine the connections
        # sparsity_mask = (torch.rand(reservoir_size, input_dim) < sparsity)
        # W_in = W_in * sparsity_mask
            
        return W_in	