#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by:  Jaideep Pathak, University of Maryland
                Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
from scipy import sparse as sparse
import os
import sys

import torch
from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from plotting_utils import *
from global_utils import *

import pickle
import time
from esn_model import ESNModel

# MEMORY TRACKING
import psutil

class ESN(object):
    def delete(self):
        del self
        
    def __init__(self, params):
        self.display_output = params["display_output"]

        self.experiment_mode = params["mode"]
        self.esn_mode = None

        self.gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu else "cpu")
        if self.display_output: self.printGPUInfo()

        if self.gpu:
            if self.display_output == True: 
                print("USING CUDA AND GPU.")
            torch.set_default_device("cuda")  # Set the default device to GPU
            torch.set_default_dtype(torch.float64)  # Set default dtype if needed
            if self.display_output == True: 
                print("# GPU MEMORY nvidia-smi in MB={:}".format(get_gpu_memory_map()))
        else:
            torch.set_default_device("cpu")  # Set the default device to CPU
            torch.set_default_dtype(torch.float64)  # Set default dtype if needed

        if self.display_output == True : print("RANDOM SEED: {:}".format(params["worker_id"]))
        self.random_seed = params["worker_id"]
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if self.gpu: torch.cuda.manual_seed(self.random_seed)

        self.class_name = params["model_name"]
        self.worker_id = params["worker_id"]
        self.input_dim = params["RDIM"]
        self.N_train = params["N_train"]
        self.N_test = params["N_test"]
        self.approx_reservoir_size = params["approx_reservoir_size"]
        self.p_in = params["p_in"]
        self.radius = params["radius"]
        self.sigma_input = params["sigma_input"]
        
        self.step_size = params["step_size"]
        self.dynamics_length = int(params["dynamics_length"]/self.step_size)
        self.iterative_prediction_length = int(params["iterative_prediction_length"]/self.step_size)

        self.num_test_ICS = params["num_test_ICS"]
        self.train_data_path = params["train_data_path"]
        self.test_data_path = params["test_data_path"]
        self.fig_dir = params["fig_dir"]
        self.model_dir = params["model_dir"]
        self.logfile_dir = params["logfile_dir"]
        self.write_to_log = params["write_to_log"]
        self.results_dir = params["results_dir"]
        self.saving_path = params["saving_path"]
        self.regularization = params["regularization"]
        self.scaler_tt = params["scaler"]
        self.system_name = params["system_name"]
        self.nodes_per_input = int(np.ceil(self.approx_reservoir_size/self.input_dim))
        self.reservoir_size = int(self.input_dim * self.nodes_per_input)
        self.sparsity = params["sparsity"]

        self.W_in = None
        self.W_h = None
        self.W_out = None

        ##########################################
        self.scaler = scaler(self.scaler_tt)
        self.noise_level = params["noise_level"]
        self.model_name = self.createModelName(params)
    
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


    def createModelName(self, params):
        keys = self.getKeysInModelName()
        str_ = "RNN-" + self.class_name + self.system_name
        str_ += "-SS_{:}".format(self.step_size)
        for key in keys:
            str_ += "-" + keys[key] + "_{:}".format(params[key])
        return str_
    
    def sendDataToCUDA(self, data):
        return data.cuda()

    def sendDataToCPU(self, data):
        return data.cpu()

    def getDataPathForTuning(self):
        train_data_path = global_params.project_path + f"/Data/{self.system_name}/Data/hypTuning/training_data_N{self.N_train}.pickle"
        test_data_path = global_params.project_path + f"/Data/{self.system_name}/Data/hypTuning/testing_data_N{self.N_test}.pickle"
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def train(self):
        torch.cuda.empty_cache()
        self.esn_mode = "train"
        self.start_time = time.time()
        dynamics_length = self.dynamics_length
        input_dim = self.input_dim
        N_used = int(self.N_train/ self.step_size)  # Adjust N_used based on step size

        with open(self.train_data_path, "rb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            data = pickle.load(file)
            train_input_sequence = data["train_input_sequence"]
            if self.step_size > 1:  
                # If step size is greater than 1, we need to downsample the input sequence
                train_input_sequence = train_input_sequence[::self.step_size, :]
                if self.display_output == True: print("Downsampling input sequence by a factor of {:}".format(self.step_size))

            if self.display_output == True : print("Adding noise to the training data. {:} per mille ".format(self.noise_level))
            train_input_sequence = addNoise(train_input_sequence, self.noise_level)
            N_all, dim = np.shape(train_input_sequence)
            if input_dim > dim: raise ValueError("Requested input dimension is wrong.")
            train_input_sequence = train_input_sequence[:N_used, :input_dim]
            del data
            
        if self.display_output == True : print("##Using {:}/{:} dimensions and {:}/{:} samples ##".format(input_dim, dim, N_used, N_all))
        if N_used > N_all: raise ValueError("Not enough samples in the training data.")
        
        if self.display_output == True : print("SCALING")
        train_input_sequence = self.scaler.scaleData(train_input_sequence)


        # Transform input sequence to pytorch tensor with shape [sequence_length, batch_size=1, dimension] and forward the network
        # train_input_sequence = np.expand_dims(train_input_sequence, axis=1)
        train_input_sequence = torch.from_numpy(train_input_sequence)
        self.model = ESNModel(self)

        if self.gpu:
            # SENDING THE TENSORS TO CUDA
            train_input_sequence = self.sendDataToCUDA(train_input_sequence)
            self.model = self.sendDataToCUDA(self.model)
            # self.model.sendModelToCuda()

        self.model.train(train_input_sequence, dynamics_length)

        train_input_sequence = self.sendDataToCPU(train_input_sequence)
        self.model = self.sendDataToCPU(self.model)

        if self.display_output == True : print("FINALISING WEIGHTS...")
        self.W_in = self.model.W_in.cpu().detach().numpy()
        self.W_h = self.model.W_h.cpu().detach().numpy()
        self.W_out = self.model.W_out.cpu().detach().numpy()
        
        if self.display_output == True : print("COMPUTING NUMBER OF PARAMETERS...")
        self.n_trainable_parameters = np.size(self.W_out)
        self.n_model_parameters = np.size(self.W_in) + np.size(self.W_h) + np.size(self.W_out)
        if self.display_output == True : print("Number of trainable parameters: {}".format(self.n_trainable_parameters))
        if self.display_output == True : print("Total number of parameters: {}".format(self.n_model_parameters))
        if self.display_output == True : print("SAVING MODEL...")
        torch.cuda.empty_cache()

    def setEsnModel(self):
        self.W_in = torch.from_numpy(self.W_in)
        self.W_h = torch.from_numpy(self.W_h)
        self.W_out = torch.from_numpy(self.W_out)
        if self.gpu:
            self.W_in = self.sendDataToCUDA(self.W_in)
            self.W_h = self.sendDataToCUDA(self.W_h)
            self.W_out = self.sendDataToCUDA(self.W_out)
            self.model = ESNModel(self)
            self.model = self.sendDataToCUDA(self.model)
        else:
            self.model = ESNModel(self)
    
    def detachEsnModel(self):
        if self.gpu:
            self.W_in = self.sendDataToCPU(self.W_in)
            self.W_h = self.sendDataToCPU(self.W_h)
            self.W_out = self.sendDataToCPU(self.W_out)
            self.model = self.sendDataToCPU(self.model)

    def validate(self):
        torch.cuda.empty_cache()
        self.esn_mode = "test"
        self.setEsnModel()
        self.testOnTestingSet()
        self.detachEsnModel()
        torch.cuda.empty_cache()
    
    def test(self):
        torch.cuda.empty_cache()
        self.esn_mode = "test"
        if self.loadModel()==0:
            self.setEsnModel()
            # self.testOnTrainingSet()
            self.testOnTestingSet()
            self.detachEsnModel()
            torch.cuda.empty_cache()
    
    def hyperTuning(self):
        self.getDataPathForTuning()
        self.train()
        self.validate()

    def testOnTrainingSet(self):
        if self.display_output == True: print("TEST ON TRAINING SET")
        with open(self.train_data_path, "rb") as file:
            data = pickle.load(file)
            train_input_sequence = data["train_input_sequence"][:, :self.input_dim]
            if self.step_size > 1:
                # If step size is greater than 1, we need to downsample the input sequence
                train_input_sequence = train_input_sequence[::self.step_size, :]
                if self.display_output == True: print("Downsampling input sequence by a factor of {:}".format(self.step_size))
            sequence_length = np.shape(train_input_sequence)[0]
            testing_ic_indexes, num_test_ICS = self.getTestingInitialIndexes(sequence_length)
            self.num_test_ICS_for_training = num_test_ICS
            dt = data["dt"]
            data_mle = data["mle"]
            del data
        targets_all, predictions_all, rmse_all, rmnse_all, targets_augment_all, predictions_augment_all, \
            rmse_avg, rmnse_avg, num_accurate_pred_05_avg, num_accurate_pred_1_avg, error_freq, freq_pred, freq_true, sp_true, sp_pred, d_temp, d_geom = \
                self.predictIndexes(train_input_sequence, num_test_ICS, testing_ic_indexes, data_mle, dt, "TRAIN")
        
        for var_name in getNamesInterestingVars():
            exec("self.{:s}_TRAIN = {:s}".format(var_name, var_name))
        for var_name in getSequencesInterestingVars():
            exec("self.{:s}_TRAIN = {:s}".format(var_name, var_name))
        return 0

    def testOnTestingSet(self):
        if self.display_output == True: print("TEST ON TESTING SET")
        with open(self.test_data_path, "rb") as file:
            data = pickle.load(file)
            test_input_sequence = data["test_input_sequence"][:, :self.input_dim]
            if self.step_size > 1:
                # If step size is greater than 1, we need to downsample the input sequence
                test_input_sequence = test_input_sequence[::self.step_size, :]
                if self.display_output == True: print("Downsampling input sequence by a factor of {:}".format(self.step_size))
            sequence_length = np.shape(test_input_sequence)[0]
            testing_ic_indexes, num_test_ICS = self.getTestingInitialIndexes(sequence_length)
            # print("Number of testing ICs: {:}".format(num_test_ICS))
            self.num_test_ICS = num_test_ICS
            dt = data["dt"]
            data_mle = data["mle"]
            del data
        targets_all, predictions_all, rmse_all, rmnse_all, targets_augment_all, predictions_augment_all, \
            rmse_avg, rmnse_avg, num_accurate_pred_05_avg, num_accurate_pred_1_avg, error_freq, freq_pred, freq_true, sp_true, sp_pred, d_temp, d_geom = \
                self.predictIndexes(test_input_sequence, num_test_ICS, testing_ic_indexes, data_mle, dt, "TEST")
        
        for var_name in getNamesInterestingVars():
            exec("self.{:s}_TEST = {:s}".format(var_name, var_name))
        for var_name in getSequencesInterestingVars():
            exec("self.{:s}_TEST = {:s}".format(var_name, var_name))
        return 0
    
    def getTestingInitialIndexes(self, sequence_length):
        max_idx = sequence_length - self.iterative_prediction_length
        min_idx = self.dynamics_length
        testing_ic_indexes = np.arange(min_idx, max_idx, step = self.iterative_prediction_length)
        np.random.shuffle(testing_ic_indexes)
        if len(testing_ic_indexes) < self.num_test_ICS:
            num_test_ICS = len(testing_ic_indexes)
        else:
            num_test_ICS = self.num_test_ICS
        return testing_ic_indexes, num_test_ICS

    def predictIndexes(self, input_sequence, num_test_ICS, ic_indexes, data_mle, dt, set_name):
        input_sequence = self.scaler.scaleData(input_sequence, reuse=1)     # reuse the statistcs of the traning sequence 
        predictions_all = []
        targets_all = []
        targets_augment_all = []
        predictions_augment_all = []
        rmse_all = []
        rmnse_all = []
        num_accurate_pred_05_all = []
        num_accurate_pred_1_all = []
        d_geom_all = []

        for ic_num in range(num_test_ICS):
            if self.display_output == True:
                print("IC {:}/{:}, {:2.3f}%".format((ic_num+1), num_test_ICS, (ic_num+1)/num_test_ICS*100))
                
            ic_idx = ic_indexes[ic_num]
            input_sequence_ic = input_sequence[ic_idx-self.dynamics_length:ic_idx+self.iterative_prediction_length]
            
            # Transform input sequence to pytorch tensor with shape [sequence_length, batch_size=1, dimension] and forward the network
            input_sequence_ic = torch.from_numpy(input_sequence_ic)
            if self.gpu:
                # SENDING THE TENSORS TO CUDA
                input_sequence_ic = self.sendDataToCUDA(input_sequence_ic)

            prediction, target, prediction_augment, target_augment, d_geom = self.model.predictSequence(input_sequence_ic, self.dynamics_length, self.iterative_prediction_length)
            if self.gpu:
                prediction= prediction.cpu().detach()
                target = target.cpu().detach()
                prediction_augment = prediction_augment.cpu().detach()
                target_augment = target_augment.cpu().detach()
                d_geom = d_geom.cpu().detach()
            prediction= prediction.numpy()
            target = target.numpy()
            prediction_augment = prediction_augment.numpy()
            target_augment = target_augment.numpy()
            d_geom = d_geom.numpy()

            if self.display_output == True: print("SEQUENCE PREDICTED...\n")
            prediction = self.scaler.descaleData(prediction)
            target = self.scaler.descaleData(target)
            prediction_augment = self.scaler.descaleData(prediction_augment)
            target_augment = self.scaler.descaleData(target_augment)
            rmse, rmnse, num_accurate_pred_05, num_accurate_pred_1, abserror = computeErrors(target, prediction, self.scaler.data_std, data_mle, dt)
            predictions_all.append(prediction)
            targets_all.append(target)
            rmse_all.append(rmse)
            rmnse_all.append(rmnse)
            targets_augment_all.append(target_augment)
            predictions_augment_all.append(prediction_augment)
            num_accurate_pred_05_all.append(num_accurate_pred_05)
            num_accurate_pred_1_all.append(num_accurate_pred_1)
            d_geom_all.append(d_geom)

        targets_all = np.array(targets_all)
        predictions_all = np.array(predictions_all)
        rmse_all = np.array(rmse_all)
        rmnse_all = np.array(rmnse_all)
        targets_augment_all = np.array(targets_augment_all)
        predictions_augment_all = np.array(predictions_augment_all)
        num_accurate_pred_05_all = np.array(num_accurate_pred_05_all)
        num_accurate_pred_1_all = np.array(num_accurate_pred_1_all)
        d_geom_all = np.array(d_geom_all)
        # print(d_geom_all)

        if self.display_output == True : print("TRAJECTORIES SHAPES:")
        if self.display_output == True : print(np.shape(targets_all))
        if self.display_output == True : print(np.shape(predictions_all))
        rmse_avg = np.mean(rmse_all)
        rmnse_avg = np.mean(rmnse_all)
        # print(rmnse_all)
        if self.display_output == True : print("AVERAGE RMNSE ERROR: {:}".format(rmnse_avg))
        num_accurate_pred_05_avg = np.mean(num_accurate_pred_05_all)
        if self.display_output == True : print("AVG NUMBER OF ACCURATE 0.5 PREDICTIONS: {:}".format(num_accurate_pred_05_avg))
        num_accurate_pred_1_avg = np.mean(num_accurate_pred_1_all)
        if self.display_output == True : print("AVG NUMBER OF ACCURATE 1 PREDICTIONS: {:}".format(num_accurate_pred_1_avg))
        freq_pred, freq_true, sp_true, sp_pred, error_freq = computeFrequencyError(targets_all, predictions_all, dt)
        if self.display_output == True : print("POWER SPECTRUM MEAN ERROR: {:}".format(error_freq))
        d_temp = temporalDistance(targets_all, predictions_all)
        if self.display_output == True : print("TEMPORAL DISTANCE: {:}".format(d_temp))
        d_geom = np.mean(d_geom_all)
        if self.display_output == True : print("GEOMETRICAL DISTANCE: {:}".format(d_geom))

        if self.display_output == True: print(f"IC INDEXES OF {set_name} PREDICTED...\n\n")
        # print(predictions_all)
        # data_path = self.saving_path +self.results_dir +self.model_name + "/1st_predicted_sequence_2.txt"
        # os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
        # np.savetxt(data_path, predictions_all[0],fmt="%.6f", delimiter=",")

        return targets_all, predictions_all, rmse_all, rmnse_all, targets_augment_all, predictions_augment_all, rmse_avg, rmnse_avg, num_accurate_pred_05_avg, num_accurate_pred_1_avg, error_freq, freq_pred, freq_true, sp_true, sp_pred, d_temp, d_geom


    def plotTestResult(self):
        os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
        # plot the test result on train and test data set
        # plotFirstThreePredictions(self, self.num_test_ICS_for_training, self.targets_all_TRAIN, self.predictions_all_TRAIN, self.rmse_all_TRAIN, self.rmnse_all_TRAIN, self.testing_ic_indexes_TRAIN, self.dt_TRAIN, self.data_mle_TRAIN, self.targets_augment_all_TRAIN, self.predictions_augment_all_TRAIN, self.dynamics_length, self.scaler.data_std, "TRAIN")
        plotFirstThreePredictions(self, self.num_test_ICS, self.targets_all_TEST, self.predictions_all_TEST, self.rmse_all_TEST, self.rmnse_all_TEST, self.testing_ic_indexes_TEST, self.dt_TEST, self.data_mle_TEST, self.targets_augment_all_TEST, self.predictions_augment_all_TEST, self.dynamics_length, self.scaler.data_std, "TEST")
        # plotSpectrum(self, self.sp_true_TRAIN, self.sp_pred_TRAIN, self.freq_true_TRAIN, self.freq_pred_TRAIN, "TRAIN")
        plotSpectrum(self, self.sp_true_TEST, self.sp_pred_TEST, self.freq_true_TEST, self.freq_pred_TEST, "TEST")

    def plotSavedResult(self):
        os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
        # plot the test result on train and test data set
        self.loadResult()
        # plotFirstThreePredictions(self, self.num_test_ICS_for_training, self.targets_all_TRAIN, self.predictions_all_TRAIN, self.rmse_all_TRAIN, self.rmnse_all_TRAIN, self.testing_ic_indexes_TRAIN, self.dt_TRAIN, self.data_mle_TRAIN, self.targets_augment_all_TRAIN, self.predictions_augment_all_TRAIN, self.dynamics_length, self.scaler.data_std, "TRAIN")
        plotFirstThreePredictions(self, self.num_test_ICS, self.targets_all_TEST, self.predictions_all_TEST, self.rmse_all_TEST, self.rmnse_all_TEST, self.testing_ic_indexes_TEST, self.dt_TEST, self.data_mle_TEST, self.targets_augment_all_TEST, self.predictions_augment_all_TEST, self.dynamics_length, self.scaler.data_std, "TEST")
        # plotSpectrum(self, self.sp_true_TRAIN, self.sp_pred_TRAIN, self.freq_true_TRAIN, self.freq_pred_TRAIN, "TRAIN")
        plotSpectrum(self, self.sp_true_TEST, self.sp_pred_TEST, self.freq_true_TEST, self.freq_pred_TEST, "TEST")

    def saveResults(self):
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
        return data
    
    def saveEvaluationResult(self):
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

    def loadModel(self):
        data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
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
            print("MODEL {:s} NOT FOUND.".format(data_path))
            return 1

    def saveModel(self):
        os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)
        os.makedirs(self.saving_path + self.model_dir + self.model_name, exist_ok=True)
        if self.display_output == True : print("Recording time...")
        self.total_training_time = time.time() - self.start_time
        if self.display_output == True : print("Total training time is {:}".format(self.total_training_time))

        if self.display_output == True : print("MEMORY TRACKING IN MB...")
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss/1024/1024
        self.memory = memory
        if self.display_output == True : print("Script used {:} MB".format(self.memory))

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
        data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
        if self.display_output == True : print("Saving the model... In path {:}\n\n".format(data_path))
        with open(data_path, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        #     del data
        # np.savetxt(self.saving_path + self.model_dir + self.model_name + "/W_in.txt", self.W_in,fmt="%.24f", delimiter=",")
        # np.savetxt(self.saving_path + self.model_dir + self.model_name + "/W_h.txt", self.W_h,fmt="%.24f", delimiter=",")
        # np.savetxt(self.saving_path + self.model_dir + self.model_name + "/W_out.txt", self.W_out,fmt="%.24f", delimiter=",")
        return 0

    