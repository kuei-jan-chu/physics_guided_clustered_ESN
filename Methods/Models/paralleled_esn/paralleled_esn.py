#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by:  Jaideep Pathak, University of Maryland
                Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import math
import numpy as np
import pickle
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv as scipypinv
import os
import sys

import torch

from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from plotting_utils import *
from global_utils import *
import pickle
import time

# MEMORY TRACKING
import psutil

from functools import partial
partial(print, flush=True)

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class ParalleledESN(object):
    def delete(self):
        del self

    def __init__(self, params):
        self.experiment_mode = params["mode"]
        self.display_output = params["display_output"]
        self.esn_mode = None
        self.worker_id = params["worker_id"]
        self.random_seed = self.worker_id + rank
        self.parallel_group_num = rank   # index for the current group

        # only use gpu to calculate geometrical distance by the first ESN
        if self.parallel_group_num == 0:
            self.gpu = torch.cuda.is_available()
            # self.gpu = False 
            self.device = torch.device("cuda" if self.gpu else "cpu")
            if self.display_output == True: self.printGPUInfo()
            if self.gpu:
                torch.set_default_device("cuda")  # Set the default device to GPU
                torch.set_default_dtype(torch.float64)  # Set default dtype if needed
                if self.display_output == True:
                    print("USING CUDA AND GPU.")
                    print("# GPU MEMORY nvidia-smi in MB={:}".format(get_gpu_memory_map()))
            else:
                torch.set_default_device("cpu")  # Set the default device to CPU
                torch.set_default_dtype(torch.float64)  # Set default dtype if needed


        # PARALLEL MODEL
        self.class_name = params["model_name"]
        self.RDIM = params["RDIM"]
        self.N_train = params["N_train"]
        self.N_test = params["N_test"]
        self.parallel_group_interaction_length = params["parallel_group_interaction_length"]
        self.parallel_group_size = params["parallel_group_size"]
        params["num_parallel_groups"] = int(params["RDIM"]/params["parallel_group_size"])   # corresponded dimensions of the input for each ESN
        self.num_parallel_groups = params["num_parallel_groups"]

        if params["RDIM"] % params["parallel_group_size"]: raise ValueError("ERROR: The parallel_group_size should divide RDIM.")
        # if size != params["num_parallel_groups"]: raise ValueError("ERROR: The num_parallel_groups is not equal to the number or ranks. Aborting...")

        if self.display_output == True and self.parallel_group_num==0: ("RANDOM SEED: {:}".format(self.random_seed))
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if self.parallel_group_num == 0 and self.gpu: torch.cuda.manual_seed(self.random_seed)

        self.input_dim =  self.parallel_group_size + params["parallel_group_interaction_length"] * 2
        self.output_dim = self.parallel_group_size

        self.approx_reservoir_size = params["approx_reservoir_size"]    # it's the total approx size for all reservoirs
        self.sparsity = params["sparsity"]
        self.radius = params["radius"]
        self.p_in = params["p_in"]
        self.sigma_input = params["sigma_input"]

        self.step_size = params["step_size"]
        self.dynamics_length = int(params["dynamics_length"]/self.step_size)
        self.iterative_prediction_length = int(params["iterative_prediction_length"]/self.step_size)

        self.num_test_ICS = params["num_test_ICS"]
        self.nodes_per_input = int(np.ceil(self.approx_reservoir_size/self.RDIM))
        self.total_reservoir_size = self.nodes_per_input * self.RDIM
        self.reservoir_size = int(np.ceil(self.total_reservoir_size/self.num_parallel_groups))

        self.regularization = params["regularization"]
        self.scaler_tt = params["scaler"]
        self.scaler = scaler(self.scaler_tt)
        self.noise_level = params["noise_level"]
        self.system_name = params["system_name"]
        self.model_name = self.createModelName(params)
        

        self.saving_path = params["saving_path"]
        self.model_dir = params["model_dir"]
        self.fig_dir = params["fig_dir"]
        self.results_dir = params["results_dir"]
        self.logfile_dir = params["logfile_dir"]
        self.write_to_log = params["write_to_log"]

        self.main_train_data_path = params["train_data_path"]
        self.main_test_data_path = params["test_data_path"]

    def printGPUInfo(self):
        print("CUDA Device available? {:}".format(torch.cuda.is_available()))
        device = torch.device('cuda' if self.gpu else 'cpu')
        print('Using device:', device)
        #Additional Info when using cuda
        if device.type == 'cuda':
            print("DEVICE NAME: {:}".format(torch.cuda.get_device_name(0)))
            print('MEMORY USAGE:')
            print('Memory allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Max memory allocated:', round(torch.cuda.max_memory_allocated(0)/1024**3,1), 'GB')
            print('Memory cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
            print('MAX memory cached:   ', round(torch.cuda.max_memory_reserved(0)/1024**3,1), 'GB')


    def getKeysInModelName(self):
        keys = {
            'RDIM':'RDIM', 
            'N_train':'N_train', 
            'N_test':'N_test',
            'approx_reservoir_size':'SIZE', 
            'sparsity':'SP', 
            'radius':'RADIUS',
            'sigma_input':'SIGMA',
            'p_in':"PIN",
            'dynamics_length':'DL',
            'noise_level':'NL',
            'iterative_prediction_length':'IPL',
            'regularization':'REG',
            'num_parallel_groups':'NUM_PARALL_GROUPS', 
            'parallel_group_size':'GS', 
            'parallel_group_interaction_length':'GIL', 
            'worker_id':'WID', 
        }
        return keys

    def createModelName(self, params):
        keys = self.getKeysInModelName()
        str_ = "RNN-" + self.class_name + "-" + self.system_name 
        if self.step_size > 1:
            str_ += "-SS_{:}".format(self.step_size)
        for key in keys:
            str_ += "-" + keys[key] + "_{:}".format(params[key])
        return str_
    
    def getReservoirWeights(self, size_x, size_y, radius, sparsity):
        # Sparse matrix with elements between 0 and 1
        if self.display_output == True and self.parallel_group_num==0: print("WEIGHT INIT")
        W_h = np.random.uniform(0,1,(size_x, size_y)) * (np.random.uniform(0,1,(size_x, size_y)) < sparsity)
        W_h = sparse.csr_matrix(W_h)
        # scale W
        if self.display_output == True and self.parallel_group_num==0: print("EIGENVALUE DECOMPOSITION")
        if size_x > 10:  # Choose a threshold based on problem size
            eigenvalues, _ = splinalg.eigs(W_h)
        else:
            eigenvalues = np.linalg.eigvals(W_h.toarray())  # Convert to dense if needed
        eigenvalues = np.abs(eigenvalues)
        W_h = (W_h/np.max(eigenvalues))*radius	# make the matrix sparse, more efficient in calculation
        return W_h

    def getInputLayerWeights(self, reservoir_size, input_dim, sigma_input, sparsity):
        W_in = np.random.uniform(-sigma_input, sigma_input,(reservoir_size, input_dim)) * (np.random.uniform(0,1,(reservoir_size, input_dim)) < sparsity)	
        W_in = sparse.csr_matrix(W_in)	# make the matrix sparse, more efficient in calculation
        # print(W_in)
        return W_in	
    

    def getTestingInitialIndexes(self, sequence_length):
        testing_ic_indexes = None  # Initialize to None
        num_test_ICS = None  # Initialize to None
        if self.parallel_group_num == 0:
            max_idx = sequence_length - self.iterative_prediction_length
            min_idx = self.dynamics_length
            testing_ic_indexes = np.arange(min_idx, max_idx, step = self.iterative_prediction_length)
            np.random.shuffle(testing_ic_indexes)
            if len(testing_ic_indexes) < self.num_test_ICS:
                num_test_ICS = len(testing_ic_indexes)
            else:
                num_test_ICS = self.num_test_ICS
            # Broadcast to all processes
        
        testing_ic_indexes = comm.bcast(testing_ic_indexes, root=0)
        num_test_ICS = comm.bcast(num_test_ICS, root=0)

        return testing_ic_indexes, num_test_ICS


    def augmentHidden(self, h):
        h_aug = h.copy()
        h_aug[::2] = h_aug[::2] ** 2.0 
        return h_aug    


    def train(self):
        self.esn_mode = "train"
        self.start_time = time.time()
        self.worker_train_data_path, self.worker_test_data_path = createParallelTrainingData(self)
        dynamics_length = self.dynamics_length
        input_dim = self.input_dim
        N_used = int(self.N_train/ self.step_size)  # Adjust N_used based on step size

        with open(self.worker_train_data_path, "rb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            data = pickle.load(file)
            train_input_sequence = data["train_input_sequence"]
            if self.step_size > 1:  
                # If step size is greater than 1, we need to downsample the input sequence
                train_input_sequence = train_input_sequence[::self.step_size, :]
                if self.display_output == True and self.parallel_group_num==0: print("Downsampling input sequence by a factor of {:}".format(self.step_size))

            if self.display_output == True and self.parallel_group_num==0: print("Adding noise to the training data. {:} per mille ".format(self.noise_level))
            train_input_sequence = addNoise(train_input_sequence, self.noise_level)            
            N_all, dim = np.shape(train_input_sequence)
            if self.input_dim > dim: raise ValueError("Requested input dimension is wrong.")
            train_input_sequence = train_input_sequence[:N_used, :input_dim]    # corresponded and adjacent dimensions in input
            dt = data["dt"]
            self.dt = dt
            del data

        if self.display_output == True and self.parallel_group_num==0: print("##Using {:}/{:} dimensions and {:}/{:} samples ##".format(input_dim, dim, N_used, N_all))
        if N_used > N_all: raise ValueError("Not enough samples in the training data.")

        if self.display_output == True and self.parallel_group_num==0: print("SCALING")
        train_input_sequence = self.scaler.scaleData(train_input_sequence)
       
        N, input_dim = np.shape(train_input_sequence)
        # Setting the reservoir size automatically to avoid overfitting
        if self.display_output == True and self.parallel_group_num==0: print("Initializing the reservoir weights...")
        W_h = self.getReservoirWeights(self.reservoir_size, self.reservoir_size, self.radius, self.sparsity)

        # Initializing the input weights
        if self.display_output == True and self.parallel_group_num==0: print("Initializing the input weights...")
        W_in = self.getInputLayerWeights(self.reservoir_size, input_dim, self.sigma_input, self.p_in)

        # TRAINING LENGTH
        tl = N - dynamics_length
        if self.display_output == True and self.parallel_group_num==0: print("\nTRAINING: Dynamics prerun...")
        # H_dyn = np.zeros((dynamics_length, 2*self.reservoir_size, 1))
        h = np.zeros((self.reservoir_size, 1))
        for t in range(dynamics_length):
            if self.display_output == True and self.parallel_group_num == 0:
                print("TRAINING - Dynamics prerun: T {:}/{:}, {:2.3f}%".format((t+1), dynamics_length, (t+1)/dynamics_length*100), end="\r")
            i = np.reshape(train_input_sequence[t], (-1,1))
            h = np.tanh(W_h @ h + W_in @ i)

        NORMEVERY = 10
        HTH = np.zeros((self.reservoir_size, self.reservoir_size))
        YTH = np.zeros((input_dim-2*self.parallel_group_interaction_length, self.reservoir_size))
        H = []
        Y = []

        if self.display_output == True and self.parallel_group_num==0: print("\nTRAINING: Teacher forcing...")
        for t in range(tl - 1):
            if self.display_output == True and self.parallel_group_num==0:
                print("TRAINING - Teacher forcing: T {:}/{:}, {:2.3f}%".format((t+1), (tl-1), (t+1)/(tl-1)*100), end="\r")
            i = np.reshape(train_input_sequence[t+dynamics_length], (-1,1))
            h = np.tanh(W_h @ h + W_in @ i)
            # AUGMENT THE HIDDEN STATE
            h_aug = self.augmentHidden(h)
            target = np.reshape(train_input_sequence[t + dynamics_length + 1, getFirstActiveIndex(self.parallel_group_interaction_length):getLastActiveIndex(self.parallel_group_interaction_length)], (-1,1))

            H.append(h_aug[:,0])
            Y.append(target[:,0])
            if (t % NORMEVERY == 0):
                # Batched approach used in the pinv case
                H = np.array(H)
                Y = np.array(Y)
                HTH += H.T @ H
                YTH += Y.T @ H
                H = []
                Y = []

        # ADDING THE REMAINNING BATCH
        if (len(H) != 0):
            H = np.array(H)
            Y = np.array(Y)
            HTH+=H.T @ H
            YTH+=Y.T @ H
            if self.display_output == True and self.parallel_group_num == 0: print("\nTEACHER FORCING ENDED.")
        
        # COMPUTING THE OUTPUT WEIGHTS
        if self.display_output == True and self.parallel_group_num==0: print("\nTRAINING: COMPUTING THE OUTPUT WEIGHTS...")
        """
        Learns mapping H -> Y with Penrose Pseudo-Inverse
        """
        I = np.identity(np.shape(HTH)[1])	
        pinv_ = scipypinv(HTH + self.regularization*I)
        W_out = YTH @ pinv_
        
        if self.display_output == True and self.parallel_group_num==0: print("FINALISING WEIGHTS...")
        self.W_in = W_in
        self.W_h = W_h
        self.W_out = W_out

        if self.display_output == True and self.parallel_group_num==0: print("COMPUTING NUMBER OF PARAMETERS...")
        self.n_trainable_parameters = np.size(self.W_out)
        self.n_model_parameters = np.size(self.W_in) + np.size(self.W_h) + np.size(self.W_out)
        
        if self.display_output == True and self.parallel_group_num==0: print("Number of trainable parameters: {}".format(self.n_trainable_parameters))
        if self.display_output == True and self.parallel_group_num==0: print("Total number of parameters: {} \n".format(self.n_model_parameters))
        if self.display_output == True and self.parallel_group_num==0: print("SAVING MODEL...")

    def predictSequence(self, input_sequence):
        W_h = self.W_h
        W_out = self.W_out
        W_in = self.W_in
        dynamics_length = self.dynamics_length
        iterative_prediction_length = self.iterative_prediction_length

        self.reservoir_size, _ = np.shape(W_h)
        N = np.shape(input_sequence)[0]
        
        # PREDICTION LENGTH
        if N != iterative_prediction_length + dynamics_length: raise ValueError("Error! N ({:}) != iterative_prediction_length + dynamics_length {:}, N={:}, iterative_prediction_length={:}, dynamics_length={:}".format(N, iterative_prediction_length+dynamics_length, N, iterative_prediction_length, dynamics_length))

        prediction_warm_up = []
        h = np.zeros((self.reservoir_size, 1))
        for t in range(dynamics_length):
            if self.display_output == True and self.parallel_group_num == 0:
                print("PREDICTION - Dynamics pre-run: T {:}/{:}, {:2.3f}%".format((t+1), dynamics_length, (t+1)/dynamics_length*100), end="\r")
            i = np.reshape(input_sequence[t], (-1,1))
            h = np.tanh(W_h @ h + W_in @ i)
            out = W_out @ self.augmentHidden(h)
            prediction_warm_up.append(out)

        target = input_sequence[dynamics_length:, getFirstActiveIndex(self.parallel_group_interaction_length):getLastActiveIndex(self.parallel_group_interaction_length)]

        prediction = []
        if self.display_output == True and self.parallel_group_num == 0:
                print("\nPREDICTION:")
        for t in range(iterative_prediction_length):
            if self.display_output == True and self.parallel_group_num == 0:
                print("PREDICTION: T {:}/{:}, {:2.3f}%".format((t+1), iterative_prediction_length, (t+1)/iterative_prediction_length*100), end="\r")
            out = W_out @ self.augmentHidden(h)
            prediction.append(out)

            # if any(math.isinf(x) for x in out):
            #     print("Infinity exists in new_input")

            # LOCAL STATE
            global_state = np.zeros((self.RDIM))
            local_state = np.zeros((self.RDIM))
            temp = np.reshape(out.copy(), (-1))
            local_state[self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = temp

            # UPDATING THE GLOBAL STATE - IMPLICIT BARRIER
            comm.Allreduce([local_state, MPI.DOUBLE], [global_state, MPI.DOUBLE], MPI.SUM)
            state_list = Circ(list(global_state.copy()))
            group_start = self.parallel_group_num * self.parallel_group_size
            group_end = group_start + self.parallel_group_size
            pgil = self.parallel_group_interaction_length
            new_input = []
            for i in range(group_start-pgil, group_end+pgil):
                new_input.append(state_list[i].copy())
            new_input = np.array(new_input)

            i = np.reshape(new_input, (-1,1)).copy()
            h = np.tanh(W_h @ h + W_in @ i)


        prediction = np.array(prediction)[:,:,0]
        prediction_warm_up = np.array(prediction_warm_up)[:,:,0]

        target_augment = input_sequence[:, getFirstActiveIndex(self.parallel_group_interaction_length):getLastActiveIndex(self.parallel_group_interaction_length)]
        prediction_augment = np.concatenate((prediction_warm_up, prediction), axis=0)

        if self.display_output == True and self.parallel_group_num == 0: print("\nSEQUENCE PREDICTED...")
        # if self.parallel_group_num == 0: print("target sequence", target)

        return prediction, target, prediction_augment, target_augment
    
    def getDataPathForTuning(self):
        train_data_path = global_params.project_path + f"/Data/{self.system_name}/Data/hypTuning/training_data_N{self.N_train}.pickle"
        test_data_path = global_params.project_path + f"/Data/{self.system_name}/Data/hypTuning/testing_data_N{self.N_test}.pickle"
        self.main_train_data_path = train_data_path
        self.main_test_data_path = test_data_path
    
    def hyperTuning(self):
        self.getDataPathForTuning()
        self.train()
        self.validate()
    
    def validate(self):
        self.esn_mode = "test"
        self.testOnTestingSet() 
        torch.cuda.empty_cache()

    def test(self):
        self.esn_mode = "test"
        if self.loadModel()==0:
            self.worker_train_data_path, self.worker_test_data_path = createParallelTrainingData(self)
            # self.testOnTrainingSet()
            self.testOnTestingSet()
            torch.cuda.empty_cache()

    def testOnTrainingSet(self):
        if self.display_output == True and self.parallel_group_num==0: print("TEST ON TRAINING SET")
        with open(self.worker_train_data_path, "rb") as file:
            data = pickle.load(file)
            train_input_sequence = data["train_input_sequence"][:,:self.input_dim]
            if self.step_size > 1:
                # If step size is greater than 1, we need to downsample the input sequence
                train_input_sequence = train_input_sequence[::self.step_size, :]
                if self.display_output == True and self.parallel_group_num==0: print("Downsampling input sequence by a factor of {:}".format(self.step_size))
            sequence_length = np.shape(train_input_sequence)[0]
            testing_ic_indexes, num_test_ICS = self.getTestingInitialIndexes(sequence_length)
            self.num_test_ICS_for_training = num_test_ICS
            # testing_ic_indexes = data["testing_ic_indexes"]
            dt = data["dt"]
            data_mle = data["mle"]
            del data
            
        targets_all, predictions_all, rmse_all, rmnse_all, targets_augment_all, predictions_augment_all, \
            rmse_avg, rmnse_avg, num_accurate_pred_05_avg, num_accurate_pred_1_avg, error_freq, freq_pred, freq_true, sp_true, sp_pred, d_temp, d_geom = \
                self.predictIndexes(train_input_sequence, num_test_ICS, testing_ic_indexes, dt, data_mle, "TRAIN")
        
        for var_name in getNamesInterestingVars():
            exec("self.{:s}_TRAIN = {:s}".format(var_name, var_name))
        for var_name in getSequencesInterestingVars():
            exec("self.{:s}_TRAIN = {:s}".format(var_name, var_name))
        return 0

    def testOnTestingSet(self):
        if self.display_output == True and self.parallel_group_num==0: print("TEST ON TESTING SET")
        with open(self.worker_test_data_path, "rb") as file:
            data = pickle.load(file)
            test_input_sequence = data["test_input_sequence"][:,:self.input_dim]
            if self.step_size > 1:
                # If step size is greater than 1, we need to downsample the input sequence
                test_input_sequence = test_input_sequence[::self.step_size, :]
                if self.display_output == True and self.parallel_group_num==0: print("Downsampling input sequence by a factor of {:}".format(self.step_size))
            sequence_length = np.shape(test_input_sequence)[0]
            testing_ic_indexes, num_test_ICS = self.getTestingInitialIndexes(sequence_length)
            self.num_test_ICS = num_test_ICS
            # testing_ic_indexes = data["testing_ic_indexes"]
            dt = data["dt"]
            data_mle = data["mle"]
            del data
            
        targets_all, predictions_all, rmse_all, rmnse_all, targets_augment_all, predictions_augment_all, \
            rmse_avg, rmnse_avg, num_accurate_pred_05_avg, num_accurate_pred_1_avg, error_freq, freq_pred, freq_true, sp_true, sp_pred, d_temp, d_geom = \
                self.predictIndexes(test_input_sequence, num_test_ICS, testing_ic_indexes, dt, data_mle, "TEST")
        
        for var_name in getNamesInterestingVars():
            exec("self.{:s}_TEST = {:s}".format(var_name, var_name))
        for var_name in getSequencesInterestingVars():
            exec("self.{:s}_TEST = {:s}".format(var_name, var_name))
        return 0

    def predictIndexes(self, input_sequence, num_test_ICS, ic_indexes, dt, data_mle, set_name):
        if self.display_output == True and self.parallel_group_num==0: print("\nPREDICTION OF INITIAL CONDITIONS")
        input_sequence = self.scaler.scaleData(input_sequence, reuse=1)
        # if self.parallel_group_num == 0:    
        #     print("train data mean", self.scaler.data_mean)
        #     print("train data std", self.scaler.data_std)
        local_predictions = []
        local_targets = []
        local_predictions_non_descaled = []
        local_targets_non_descaled = []
        local_predictions_augment = []
        local_targets_augment = []

        for ic_num in range(num_test_ICS):
            if self.display_output == True and self.parallel_group_num == 0:
                print("IC {:}/{:}, {:2.3f}%".format((ic_num+1), num_test_ICS, (ic_num+1)/num_test_ICS*100))
            
            ic_idx = ic_indexes[ic_num]
            input_sequence_ic = input_sequence[ic_idx-self.dynamics_length:ic_idx+self.iterative_prediction_length]
            prediction, target, prediction_augment, target_augment = self.predictSequence(input_sequence_ic)
            # if self.parallel_group_num == 1: print("target sequence", self.scaler.descaleData(input_sequence_ic.copy()))
            local_predictions_non_descaled.append(prediction.copy())
            local_targets_non_descaled.append(target.copy())
            prediction = self.scaler.descaleDataParallel(prediction, self.parallel_group_interaction_length)
            target = self.scaler.descaleDataParallel(target, self.parallel_group_interaction_length)
            # if self.parallel_group_num == 1: print("target sequence", target)
            prediction_augment = self.scaler.descaleDataParallel(prediction_augment,self.parallel_group_interaction_length)
            target_augment = self.scaler.descaleDataParallel(target_augment, self.parallel_group_interaction_length)
            local_predictions.append(prediction)
            local_targets.append(target)
            local_predictions_augment.append(prediction_augment)
            local_targets_augment.append(target_augment)

        local_predictions = np.array(local_predictions)
        local_targets = np.array(local_targets)
        local_predictions_non_descaled = np.array(local_predictions_non_descaled)
        local_targets_non_descaled = np.array(local_targets_non_descaled)
        local_predictions_augment = np.array(local_predictions_augment)
        local_targets_augment = np.array(local_targets_augment)
        comm.Barrier()

        predictions_all_proxy = np.zeros((num_test_ICS, self.iterative_prediction_length, self.RDIM))
        targets_all_proxy = np.zeros((num_test_ICS, self.iterative_prediction_length, self.RDIM))
        predictions_non_descaled_all_proxy = np.zeros((num_test_ICS, self.iterative_prediction_length, self.RDIM))
        targets_non_descaled_all_proxy = np.zeros((num_test_ICS, self.iterative_prediction_length, self.RDIM))
        predictions_augment_all_proxy = np.zeros((num_test_ICS, self.dynamics_length+self.iterative_prediction_length, self.RDIM))
        targets_augment_all_proxy = np.zeros((num_test_ICS, self.dynamics_length+self.iterative_prediction_length, self.RDIM))
        scaler_std_proxy = np.zeros((self.RDIM))

        # SETTING THE LOCAL VALUES
        predictions_all_proxy[:,:,self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = local_predictions
        targets_all_proxy[:,:,self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = local_targets
        predictions_non_descaled_all_proxy[:,:,self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = local_predictions_non_descaled
        targets_non_descaled_all_proxy[:,:,self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = local_targets_non_descaled
        predictions_augment_all_proxy[:,:,self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = local_predictions_augment
        targets_augment_all_proxy[:,:,self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = local_targets_augment
        scaler_std_proxy[self.parallel_group_num*self.parallel_group_size:(self.parallel_group_num+1)*self.parallel_group_size] = self.scaler.data_std[getFirstActiveIndex(self.parallel_group_interaction_length):getLastActiveIndex(self.parallel_group_interaction_length)]
        # if self.parallel_group_num == 0: print(targets_all_proxy)

        predictions_all = np.zeros((num_test_ICS, self.iterative_prediction_length, self.RDIM)) if(self.parallel_group_num == 0) else None
        targets_all = np.zeros((num_test_ICS, self.iterative_prediction_length, self.RDIM)) if(self.parallel_group_num == 0) else None
        predictions_non_descaled_all = np.zeros((num_test_ICS, self.iterative_prediction_length, self.RDIM)) if(self.parallel_group_num == 0) else None
        targets_non_descaled_all = np.zeros((num_test_ICS, self.iterative_prediction_length, self.RDIM)) if(self.parallel_group_num == 0) else None
        predictions_augment_all = np.zeros((num_test_ICS, self.dynamics_length+self.iterative_prediction_length, self.RDIM)) if(self.parallel_group_num == 0) else None
        targets_augment_all = np.zeros((num_test_ICS, self.dynamics_length+self.iterative_prediction_length, self.RDIM)) if(self.parallel_group_num == 0) else None
        scaler_std = np.zeros((self.RDIM)) if(self.parallel_group_num == 0) else None

        comm.Reduce([predictions_all_proxy, MPI.DOUBLE], [predictions_all, MPI.DOUBLE], MPI.SUM, root=0)
        comm.Reduce([targets_all_proxy, MPI.DOUBLE], [targets_all, MPI.DOUBLE], MPI.SUM, root=0)
        comm.Reduce([predictions_non_descaled_all_proxy, MPI.DOUBLE], [predictions_non_descaled_all, MPI.DOUBLE], MPI.SUM, root=0)
        comm.Reduce([targets_non_descaled_all_proxy, MPI.DOUBLE], [targets_non_descaled_all, MPI.DOUBLE], MPI.SUM, root=0)
        comm.Reduce([predictions_augment_all_proxy, MPI.DOUBLE], [predictions_augment_all, MPI.DOUBLE], MPI.SUM, root=0)
        comm.Reduce([targets_augment_all_proxy, MPI.DOUBLE], [targets_augment_all, MPI.DOUBLE], MPI.SUM, root=0)
        comm.Reduce([scaler_std_proxy, MPI.DOUBLE], [scaler_std, MPI.DOUBLE], MPI.SUM, root=0)
        self.scaler.data_std_for_all_reservoirs = scaler_std if(self.parallel_group_num == 0) else None
        # if self.parallel_group_num == 0: print(targets_all)

        if self.display_output == True and self.parallel_group_num==0: print("PREDICTION OF INITIAL CONDITIONS FINISHED")
        if(self.parallel_group_num == 0):
            if self.display_output == True: print("MASTER RANK GATHERING PREDICTIONS...")
            # COMPUTING OTHER QUANTITIES
            rmse_all = []
            rmnse_all = []
            num_accurate_pred_05_all = []
            num_accurate_pred_1_all = []
            for ic_num in range(num_test_ICS):
                prediction = predictions_all[ic_num]
                target = targets_all[ic_num]
                rmse, rmnse, num_accurate_pred_05, num_accurate_pred_1, abserror = computeErrors(target, prediction, scaler_std, data_mle, dt)
                rmse_all.append(rmse)
                rmnse_all.append(rmnse)
                num_accurate_pred_05_all.append(num_accurate_pred_05)
                num_accurate_pred_1_all.append(num_accurate_pred_1)

            rmse_all = np.array(rmse_all)
            rmnse_all = np.array(rmnse_all)
            num_accurate_pred_05_all = np.array(num_accurate_pred_05_all)
            num_accurate_pred_1_all = np.array(num_accurate_pred_1_all)

            if self.display_output == True: print("TRAJECTORIES SHAPES:")
            if self.display_output == True: print(np.shape(targets_all))
            if self.display_output == True: print(np.shape(predictions_all))

            rmse_avg = np.mean(rmse_all)
            rmnse_avg = np.mean(rmnse_all)
            if self.display_output == True: print("AVERAGE RMNSE ERROR: {:}".format(rmnse_avg))
            num_accurate_pred_05_avg = np.mean(num_accurate_pred_05_all)
            if self.display_output == True: print("AVG NUMBER OF ACCURATE 0.5 PREDICTIONS: {:}".format(num_accurate_pred_05_avg))
            num_accurate_pred_1_avg = np.mean(num_accurate_pred_1_all)
            if self.display_output == True: print("AVG NUMBER OF ACCURATE 1 PREDICTIONS: {:}".format(num_accurate_pred_1_avg))
            freq_pred, freq_true, sp_true, sp_pred, error_freq = computeFrequencyError(targets_all, predictions_all, dt)
            if self.display_output == True: print("POWER SPECTRUM MEAN ERROR: {:}".format(error_freq))
            d_temp = temporalDistance(targets_all, predictions_all)
            if self.display_output == True : print("TEMPORAL DISTANCE: {:}".format(d_temp))
            d_geom = calculateGeometricDistanceForBatchData(targets_non_descaled_all, predictions_non_descaled_all, self.gpu)
            if self.display_output == True : print("GEOMETRICAL DISTANCE: {:}".format(d_geom))

            # data_path = self.saving_path +self.results_dir +self.model_name + "/1st_test_sequence.txt"
            # os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
            # np.savetxt(data_path, targets_all[0],fmt="%.6f", delimiter=",")
        else:
            rmse_all, rmnse_all, rmse_avg, rmnse_avg, num_accurate_pred_05_avg, num_accurate_pred_1_avg, error_freq, predictions_all, targets_all, freq_pred, freq_true, sp_true, sp_pred, d_temp, d_geom = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        if self.display_output == True and self.parallel_group_num==0: print(f"IC INDEXES OF {set_name} PREDICTED...\n\n")
        return targets_all, predictions_all, rmse_all, rmnse_all, targets_augment_all, predictions_augment_all, rmse_avg, rmnse_avg, num_accurate_pred_05_avg, num_accurate_pred_1_avg, error_freq, freq_pred, freq_true, sp_true, sp_pred, d_temp, d_geom


    def plotTestResult(self):
        if(self.parallel_group_num == 0):
            os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
            # plot the test result on train and test data set
            # plotFirstThreePredictions(self, self.num_test_ICS_for_training, self.targets_all_TRAIN, self.predictions_all_TRAIN, self.rmse_all_TRAIN, self.rmnse_all_TRAIN, self.testing_ic_indexes_TRAIN, self.dt_TRAIN, self.data_mle_TRAIN, self.targets_augment_all_TRAIN, self.predictions_augment_all_TRAIN, self.dynamics_length, self.scaler.data_std_for_all_reservoirs, "TRAIN")
            plotFirstThreePredictions(self, self.num_test_ICS, self.targets_all_TEST, self.predictions_all_TEST, self.rmse_all_TEST, self.rmnse_all_TEST, self.testing_ic_indexes_TEST, self.dt_TEST, self.data_mle_TEST, self.targets_augment_all_TEST, self.predictions_augment_all_TEST, self.dynamics_length, self.scaler.data_std_for_all_reservoirs, "TEST")
            # plotSpectrum(self, self.sp_true_TRAIN, self.sp_pred_TRAIN, self.freq_true_TRAIN, self.freq_pred_TRAIN, "TRAIN")
            plotSpectrum(self, self.sp_true_TEST, self.sp_pred_TEST, self.freq_true_TEST, self.freq_pred_TEST, "TEST")

    def plotSavedResult(self):
        if(self.parallel_group_num == 0):
            os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
            # plot the test result on train and test data set
            self.loadResult()
            # plotFirstThreePredictions(self, self.num_test_ICS_for_training, self.targets_all_TRAIN, self.predictions_all_TRAIN, self.rmse_all_TRAIN, self.rmnse_all_TRAIN, self.testing_ic_indexes_TRAIN, self.dt_TRAIN, self.data_mle_TRAIN, self.targets_augment_all_TRAIN, self.predictions_augment_all_TRAIN, self.dynamics_length, self.scaler.data_std_for_all_reservoirs, "TRAIN")
            plotFirstThreePredictions(self, self.num_test_ICS, self.targets_all_TEST, self.predictions_all_TEST, self.rmse_all_TEST, self.rmnse_all_TEST, self.testing_ic_indexes_TEST, self.dt_TEST, self.data_mle_TEST, self.targets_augment_all_TEST, self.predictions_augment_all_TEST, self.dynamics_length, self.scaler.data_std_for_all_reservoirs, "TEST")
            # plotSpectrum(self, self.sp_true_TRAIN, self.sp_pred_TRAIN, self.freq_true_TRAIN, self.freq_pred_TRAIN, "TRAIN")
            plotSpectrum(self, self.sp_true_TEST, self.sp_pred_TEST, self.freq_true_TEST, self.freq_pred_TEST, "TEST")

    def saveResults(self):
        if self.parallel_group_num == 0: 
            os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)
            os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
            if self.write_to_log == 1:
                logfile_test = self.saving_path + self.logfile_dir + self.model_name  + "/{:}_test.txt".format(self.experiment_mode)
                writeToTestLogFile(logfile_test, self)
            data = {}
            data["model_name"] = self.model_name

            ## used for plotting saved results
            data["num_test_ICS"] = self.num_test_ICS
            data["num_test_ICS_for_training"] = self.num_test_ICS_for_training
            data["scaler"] = self.scaler    
            
            for var_name in getNamesInterestingVars():
                exec("data['{:s}_TEST'] = self.{:s}_TEST".format(var_name, var_name))
                # exec("data['{:s}_TRAIN'] = self.{:s}_TRAIN".format(var_name, var_name))
            data_path = self.saving_path + self.results_dir + self.model_name + "/evaluation_results.pickle"
            with open(data_path, "wb") as file:
                # Pickle the "data" dictionary using the highest protocol available.
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

            for var_name in getSequencesInterestingVars():
                exec("data['{:s}_TEST'] = self.{:s}_TEST".format(var_name, var_name))
                # exec("data['{:s}_TRAIN'] = self.{:s}_TRAIN".format(var_name, var_name))
            data_path = self.saving_path + self.results_dir + self.model_name + "/results.pickle"
            with open(data_path, "wb") as file:
                # Pickle the "data" dictionary using the highest protocol available.
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
                del data
            return 0

    def loadResult(self):
        # Load the results data
        data_path = self.saving_path +self.results_dir +self.model_name + "/results.pickle"
        with open(data_path, "rb") as file:
            data = pickle.load(file)
            # load result on train/test data
            for var_name in getNamesInterestingVars():
                # exec("self.{:s}_TRAIN = data['{:s}_TRAIN']".format(var_name, var_name))
                exec("self.{:s}_TEST = data['{:s}_TEST']".format(var_name, var_name))
            for var_name in getSequencesInterestingVars():
                # exec("self.{:s}_TRAIN = data['{:s}_TRAIN']".format(var_name, var_name))
                exec("self.{:s}_TEST = data['{:s}_TEST']".format(var_name, var_name))
            self.scaler = data["scaler"]    ## used for plotting saved results
            self.num_test_ICS =  data["num_test_ICS"] 
            self.num_test_ICS_for_training = data["num_test_ICS_for_training"] 
            del data
        return 0
        
    def saveEvaluationResult(self):
        if self.parallel_group_num == 0: 
            os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)
            os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
            if self.write_to_log == 1:
                logfile_test = self.saving_path + self.logfile_dir + self.model_name  + "/{:}_test.txt".format(self.experiment_mode)
                writeToTestLogFile(logfile_test, self)
                
            data = {}
            data["model_name"] = self.model_name
            data["num_test_ICS"] = self.num_test_ICS
            for var_name in getNamesInterestingVars():
                # exec("data['{:s}_TRAIN'] = self.{:s}_TRAIN".format(var_name, var_name))
                exec("data['{:s}_TEST'] = self.{:s}_TEST".format(var_name, var_name))
            data_path = self.saving_path + self.results_dir + self.model_name + "/evaluation_results.pickle"
            with open(data_path, "wb") as file:
                # Pickle the "data" dictionary using the highest protocol available.
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
                del data
            return 0
    
    def loadEvaluationResult(self):
        # Load the results data
        data_path = self.saving_path +self.results_dir +self.model_name + "/evaluation_results.pickle"
        with open(data_path, "rb") as file:
            data = pickle.load(file)
            # load result on train/test data
            for var_name in getNamesInterestingVars():
                # exec("self.{:s}_TRAIN = data['{:s}_TRAIN']".format(var_name, var_name))
                exec("self.{:s}_TEST = data['{:s}_TEST']".format(var_name, var_name))
        return data
    
    def saveValidationResult(self):
        if self.parallel_group_num == 0: 
            os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)
            os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
            if self.write_to_log == 1:
                logfile_test = self.saving_path + self.logfile_dir + self.model_name  + "/{:}_test.txt".format(self.experiment_mode)
                writeToValidationLogFile(logfile_test, self)
                
            data = {}
            data["model_name"] = self.model_name
            data["num_test_ICS"] = self.num_test_ICS
            for var_name in getNamesInterestingVars():
                exec("data['{:s}_TEST'] = self.{:s}_TEST".format(var_name, var_name))
            data_path = self.saving_path + self.results_dir + self.model_name + "/validation_results.pickle"
            with open(data_path, "wb") as file:
                # Pickle the "data" dictionary using the highest protocol available.
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
                del data
            return 0
    
    def loadValidationResult(self):
        # Load the results data
        data_path = self.saving_path +self.results_dir +self.model_name + "/validation_results.pickle"
        with open(data_path, "rb") as file:
            data = pickle.load(file)
            # load result on train/test data
            for var_name in getNamesInterestingVars():
                exec("self.{:s}_TEST = data['{:s}_TEST']".format(var_name, var_name))
        return data

    def saveModel(self):
        os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)
        os.makedirs(self.saving_path + self.model_dir + self.model_name, exist_ok=True)
        if self.display_output == True and self.parallel_group_num==0: print("Recording time...")
        self.total_training_time = time.time() - self.start_time
        if self.display_output == True and self.parallel_group_num==0: print("Total training time is {:}".format(self.total_training_time))

        if self.display_output == True and self.parallel_group_num==0: print("MEMORY TRACKING IN MB...")
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss/1024/1024
        self.memory = memory
        if self.display_output == True and self.parallel_group_num==0: print("Script used {:} MB".format(self.memory))

        if self.write_to_log == 1:
            logfile_train = self.saving_path + self.logfile_dir + self.model_name  + "/{:}_train.txt".format(self.experiment_mode)
            writeToTrainLogFile(logfile_train, self)

        data = {
        "memory":self.memory,
        "n_trainable_parameters":self.n_trainable_parameters,
        "n_model_parameters":self.n_model_parameters,
        "total_training_time":self.total_training_time,
        "W_out":self.W_out,
        "W_in":self.W_in,
        "W_h":self.W_h,
        "scaler":self.scaler,
        }
        data_path = self.saving_path + self.model_dir + self.model_name + "/data_member_{:d}.pickle".format(self.parallel_group_num)
        if self.display_output == True and self.parallel_group_num==0: print("Saving the model... In path {:}\n\n".format(data_path))
        with open(data_path, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data
        return 0
    
    def loadModel(self):
        data_path = self.saving_path + self.model_dir + self.model_name + "/data_member_{:d}.pickle".format(self.parallel_group_num)
        try:
            with open(data_path, "rb") as file:
                data = pickle.load(file)
                self.W_out = data["W_out"]
                self.W_in = data["W_in"]
                self.W_h = data["W_h"]
                self.scaler = data["scaler"]
                del data
            return 0
        except:
            if self.display_output == True: print("MODEL {:s} NOT FOUND.".format(data_path))
            return 1
        