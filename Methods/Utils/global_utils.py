#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import subprocess
import sys
# from typing import Required
# from js2py import require
# from traitlets import default
import numpy as np
import pickle
import io
import os
import torch as tc
from scipy.stats import sem  # Import SEM calculation
from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from hyper_tuning_utils import *
from plotting_utils import *


def getNamesInterestingVars():
    # THE MODEL SHOULD OUTPUT THE FOLLOWING VARIABLES:
    var_names = [
        'testing_ic_indexes',
        'dt',
        'data_mle',
        'rmnse_all', 
        'rmnse_avg',
        'rmse_avg',
        'num_accurate_pred_05_avg',
        'num_accurate_pred_1_avg',
        'error_freq',
        'freq_pred',
        'freq_true',
        'sp_true',
        'sp_pred',
        'd_temp', 
        'd_geom',
    ]
    return var_names

def getSequencesInterestingVars():
    # THE MODEL SHOULD OUTPUT THE FOLLOWING VARIABLES:
    var_names = [
        'targets_all', 
        'predictions_all', 
        'hidden_state_all',
        'targets_augment_all', 
        'predictions_augment_all',
        'rmse_all',
    ]
    return var_names

def writeToTrainLogFile(logfile_train, model):
    with io.open(logfile_train, 'a+') as f:
        f.write("model_name:" + str(model.model_name) \
            + "mode:" + str(model.esn_mode) \
            + ":memory:" +"{:.2f}".format(model.memory) \
            + ":total_training_time:" + "{:.2f}".format(model.total_training_time) \
            + ":n_model_parameters:" + str(model.n_model_parameters) \
            + ":n_trainable_parameters:" + str(model.n_trainable_parameters) \
            + "\n"
            )
    return 0

def writeToTestLogFile(logfile_test, model):
    with io.open(logfile_test, 'a+') as f:
        f.write("model_name:" + str(model.model_name) \
            + "mode:" + str(model.esn_mode) \
            + ":num_test_ICS:" + "{:.2f}".format(model.num_test_ICS) \
            + ":num_accurate_pred_05_avg_TEST:" + "{:.2f}".format(model.num_accurate_pred_05_avg_TEST) \
            + ":num_accurate_pred_1_avg_TEST:" + "{:.2f}".format(model.num_accurate_pred_1_avg_TEST) \
            + ":num_accurate_pred_05_avg_TRAIN:" + "{:.2f}".format(model.num_accurate_pred_05_avg_TRAIN) \
            + ":num_accurate_pred_1_avg_TRAIN:" + "{:.2f}".format(model.num_accurate_pred_1_avg_TRAIN) \
            + ":error_freq_TRAIN:" + "{:.2f}".format(model.error_freq_TRAIN) \
            + ":error_freq_TEST:" + "{:.2f}".format(model.error_freq_TEST) \
            + ":d_geom_TEST:" + "{:.2f}".format(model.d_geom_TEST) \
            + ":d_temp_TRAIN:" + "{:.2f}".format(model.d_temp_TRAIN) \
            + ":d_temp_TEST:" + "{:.2f}".format(model.d_temp_TEST) \
            + "\n"
            )
    return 0

def writeToValidationLogFile(logfile_test, model):
    with io.open(logfile_test, 'a+') as f:
        f.write("model_name:" + str(model.model_name) \
            + "mode:" + str(model.esn_mode) \
            + ":num_test_ICS:" + "{:.2f}".format(model.num_test_ICS) \
            + ":num_accurate_pred_05_avg_TEST:" + "{:.2f}".format(model.num_accurate_pred_05_avg_TEST) \
            + ":num_accurate_pred_1_avg_TEST:" + "{:.2f}".format(model.num_accurate_pred_1_avg_TEST) \
            + ":error_freq_TEST:" + "{:.2f}".format(model.error_freq_TEST) \
            + ":d_geom_TEST:" + "{:.2f}".format(model.d_geom_TEST) \
            + ":d_temp_TEST:" + "{:.2f}".format(model.d_temp_TEST) \
            + "\n"
            )
    return 0


def replaceNaN(data):
    data[np.isnan(data)]=float('Inf')
    return data


def addNoise(data, percent):
    std_data = np.std(data, axis=0)
    std_data = np.reshape(std_data, (1, -1))
    std_data = np.repeat(std_data, np.shape(data)[0], axis=0)
    noise = np.multiply(np.random.randn(*np.shape(data)), percent/1000.0*std_data)
    data += noise
    return data

class scaler(object):
    def __init__(self, tt):
        self.tt = tt
        self.data_min = 0
        self.data_max = 0
        self.data_mean = 0
        self.data_std = 0       

    def scaleData(self, input_sequence, reuse=None):
        ## reuse means use the min, max, mean, std values of the training data set
        if reuse == None:
            self.data_mean = np.mean(input_sequence,0)
            self.data_std = np.std(input_sequence,0)
            self.data_min = np.min(input_sequence,0)
            self.data_max = np.max(input_sequence,0)
        if self.tt == "MinMaxZeroOne":
            input_sequence = np.array((input_sequence-self.data_min)/(self.data_max-self.data_min))
        elif self.tt == "Standard":
            safe_std = np.where(self.data_std == 0, 1, self.data_std)
            input_sequence = np.array((input_sequence-self.data_mean)/safe_std)
        elif self.tt != "no":
            raise ValueError("Scaler not implemented.")
        return input_sequence

    def descaleData(self, input_sequence):
        if self.tt == "MinMaxZeroOne":
            input_sequence = np.array(input_sequence*(self.data_max - self.data_min) + self.data_min)
        elif self.tt == "Standard":
            input_sequence = np.array(input_sequence*self.data_std + self.data_mean)
        elif self.tt != "no":
            raise ValueError("Scaler not implemented.")
        return input_sequence

    def descaleDataParallel(self, input_sequence, interaction_length):
        # Descaling in the parallel model requires to substract the neighboring points from the scaler
        if self.tt == "MinMaxZeroOne":
            input_sequence = np.array(input_sequence*(self.data_max[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)] - self.data_min[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)]) + self.data_min[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)])
        elif self.tt == "Standard":
            input_sequence = np.array(input_sequence*self.data_std[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)] + self.data_mean[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)])
        elif self.tt != "no":
            raise ValueError("Scaler not implemented.")
        return input_sequence


def computeErrors(target, prediction, std, data_mle, dt):
    prediction = replaceNaN(prediction)
    # ABOSOLUTE ERROR
    abserror = np.mean(np.abs(target-prediction), axis=1)
    # SQUARE ERROR
    serror = np.square(target-prediction)
    # MEAN (over-space) SQUARE ERROR
    mse = np.mean(serror, axis=1)
    # ROOT MEAN SQUARE ERROR
    rmse = np.sqrt(mse)
    # NORMALIZED SQUARE ERROR
    nserror = serror/np.square(std)
    # MEAN (over-space) NORMALIZED SQUARE ERROR
    mnse = np.mean(nserror, axis=1)
    # ROOT MEAN NORMALIZED SQUARE ERROR
    rmnse = np.sqrt(mnse)
    num_accurate_pred_05 = getNumberOfAccuratePredictions(rmnse, data_mle, dt, 0.5)
    num_accurate_pred_1 = getNumberOfAccuratePredictions(rmnse, data_mle, dt, 1)
    return rmse, rmnse, num_accurate_pred_05, num_accurate_pred_1, abserror

def getNumberOfAccuratePredictions(nerror, data_mle, dt, tresh=0.05):
    nerror_bool = nerror < tresh
    n_max = np.shape(nerror)[0]
    n = 0
    while nerror_bool[n] == True:
        n += 1
        if n == n_max: break
    if data_mle > 1e-2:
        valid_time = n * data_mle * dt
    else:
        valid_time = n * dt
    # valid_time = n * dt / data_mle
    return valid_time


def computeFrequencyError(targets_all, predictions_all, dt):
    sequence_length = predictions_all.shape[1]
    t_washout = int(0.25*sequence_length)
    sp_pred, freq_pred = computeSpectrum(predictions_all[:,t_washout:,:], dt)
    sp_true, freq_true = computeSpectrum(targets_all[:,t_washout:,:], dt)
    # sp_pred, freq_pred = computeSpectrum(predictions_all, dt)
    # sp_true, freq_true = computeSpectrum(targets_all, dt)
    error_freq = np.mean(np.abs(sp_pred - sp_true))
    return freq_pred, freq_true, sp_true, sp_pred, error_freq

def computeSpectrum(data_all, dt):
    # Of the shape [n_ics, T, n_dim]
    spectrum_db = []
    for data in data_all:
        data = np.transpose(data)
        for d in data:
            freq, s_dbfs = dbfft(d, 1/dt)
            spectrum_db.append(s_dbfs)
    spectrum_db = np.array(spectrum_db).mean(axis=0)
    return spectrum_db, freq


def dbfft(x, fs):
    """
    Calculate spectrum in dB scale
    Args:
        x: input signal
        fs: sampling frequency
    Returns:
        freq: frequency vector
        s_db: spectrum in dB scale
    """
    N = len(x)  # Length of input sequence
    if N % 2 != 0:
        x = x[:-1]
        N = len(x)
    x = np.reshape(x, (1,N))
    # Calculate real FFT and frequency vector
    sp = np.fft.rfft(x)
    freq = np.arange((N / 2) + 1) / (float(N) / fs)
    # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum.
    s_mag = np.abs(sp) * 2 / N
    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag)
    s_dbfs = s_dbfs[0]
    return freq, s_dbfs


def gauss(x, sigma=1):
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-1 / 2 * (x / sigma) ** 2)

def get_kernel(sigma):
    size = sigma * 10 + 1
    kernel = list(range(size))
    kernel = [float(k) - int(size / 2) for k in kernel]
    kernel = [gauss(k, sigma) for k in kernel]
    kernel = [k / np.sum(kernel) for k in kernel]
    return kernel

def kernel_smoothen(data, kernel_sigma=1):
    """
    Smoothen data with Gaussian kernel
    @param kernel_sigma: standard deviation of gaussian, kernel_size is adapted to that
    @return: internal data is modified but nothing returned
    """
    kernel = get_kernel(kernel_sigma)
    data_final = data.copy()
    data_conv = np.convolve(data[:], kernel)
    pad = int(len(kernel) / 2)
    data_final[:] = data_conv[pad:-pad]
    data = data_final
    return data

def get_average_spectrum(trajectories, smoothing):
    '''average trajectories to fulfill conditions for the application
    of the Hellinger distance '''
    spectrum = []
    for trajectory in trajectories:
        trajectory = (trajectory-trajectory.mean())/trajectory.std()

        n = len(trajectory)
        if n % 2 != 0:
            trajectory = trajectory[:-1]
            n = len(trajectory)
        trajectory = np.reshape(trajectory, (n,))

        fft_real = np.fft.rfft(trajectory,norm='ortho')
        fft_magnitude = np.abs(fft_real)**2 * 2 / n
        fft_smoothed = kernel_smoothen(fft_magnitude, kernel_sigma=smoothing)
        spectrum.append(fft_smoothed)
    spectrum = np.nanmean(np.array(spectrum),axis=0)
    return spectrum

def temporalDistance(targets_all, predictions_all):
    SMOOTHING_SIGMA = 2
    FREQUENCY_CUTOFF = 500
    sequence_length = targets_all.shape[1]
    t_washout = int(0.25*sequence_length)

    dim_nums = predictions_all.shape[2]
    ps_distance_per_dim = []
    for dim in range(dim_nums):
        spectrum_true = get_average_spectrum(targets_all[:, t_washout:, dim], SMOOTHING_SIGMA)
        spectrum_gen = get_average_spectrum(predictions_all[:, t_washout:, dim], SMOOTHING_SIGMA)
        # spectrum_true = get_average_spectrum(targets_all[:, :, dim], SMOOTHING_SIGMA)
        # spectrum_gen = get_average_spectrum(predictions_all[:, :, dim], SMOOTHING_SIGMA)
        # print(len(spectrum_gen))
        spectrum_true = spectrum_true[:FREQUENCY_CUTOFF]
        spectrum_gen = spectrum_gen[:FREQUENCY_CUTOFF]
        BC = np.trapz(np.sqrt(spectrum_true*spectrum_gen))
        # BC = np.sum(np.sqrt(spectrum_true*spectrum_gen))
        hellinger_dist = np.sqrt(1-BC)

        ps_distance_per_dim.append(hellinger_dist)
    return np.array(ps_distance_per_dim).mean(axis=0)


def evaluateLikelihoodGmmForDiagonalCovariance(z, mu, std):
    T = mu.shape[0]
    mu = mu.reshape((1, T, -1))

    vec = z - mu  # calculate difference for every time step
    vec=vec.float()
    precision = 1 / (std ** 2)
    precision = tc.diag_embed(precision).float()
    prec_vec = tc.einsum('zij,azj->azi', precision, vec)
    exponent = tc.einsum('abc,abc->ab', vec, prec_vec)
    sqrt_det_of_cov = tc.prod(std, dim=1)
    likelihood = tc.exp(-0.5 * exponent) / sqrt_det_of_cov
    return likelihood.sum(dim=1) / T


def calculateKLMonteCarlo(mu_gen, cov_gen, mu_inf, cov_inf):
    mc_n = 1000
    t = tc.randint(0, mu_inf.shape[0], (mc_n,))

    std_inf = tc.sqrt(cov_inf)
    std_gen = tc.sqrt(cov_gen)

    z_sample = (mu_gen[t] + std_gen[t] * tc.randn(mu_gen[t].shape)).reshape((mc_n, 1, -1))

    prior = evaluateLikelihoodGmmForDiagonalCovariance(z_sample, mu_gen, std_gen)
    posterior = evaluateLikelihoodGmmForDiagonalCovariance(z_sample, mu_inf, std_inf)

    # clean from outliers
    nonzeros = (prior != 0)
    if any(prior == 0):
        prior = prior[nonzeros]
        posterior = posterior[nonzeros]
    outlier_ratio = (1 - nonzeros.float()).mean()

    # kl_mc = tc.mean(tc.log(prior + 1e-8) - tc.log(posterior + 1e-8) , dim=0)
    kl_mc = tc.mean(tc.log(prior) - tc.log(posterior) , dim=0)
    # kl_mc = tc.mean(tc.log(prior + 1e-15) - tc.log(posterior + 1e-15) , dim=0)
    # kl_mc = tc.mean(tc.log(prior + 1e-32) - tc.log(posterior + 1e-32) , dim=0)
    # print("kl after clean", kl_mc)
    return kl_mc, outlier_ratio


def calculateGeometricDistance(target, prediction):
    # print("target", target)
    # print("prediction", prediction)
    sequence_length, dim = target.shape
    if dim > 40:
        return tc.tensor(tc.inf)
    time_end = min(sequence_length, 10000)
    t_washout = int(0.25*sequence_length)
    time_steps = time_end - t_washout

    mu_gen= target[t_washout:time_end]
    mu_inf= prediction[t_washout:time_end]

    # print("mu_gen", mu_gen)
    # print("mu_inf", mu_inf)
    
    scaling = 1.
    cov_gen = tc.ones(dim).repeat(time_steps, 1)*scaling
    cov_inf = tc.ones(dim).repeat(time_steps, 1)*scaling

    kl_mc, _  = calculateKLMonteCarlo(mu_gen, cov_gen, mu_inf, cov_inf)
    return kl_mc

def calculateGeometricDistanceForBatchData(target_all, prediction_all, gpu):
    target_all = tc.from_numpy(target_all)
    prediction_all = tc.from_numpy(prediction_all)
    sequences_num = target_all.shape[0]
    sequence_length, dim = target_all[0].shape
    
    if dim > 40:
        kl_mc_avg = tc.tensor(tc.inf)
        if gpu:
            kl_mc_avg = kl_mc_avg.cpu()
        kl_mc_avg = kl_mc_avg.numpy()
        return kl_mc_avg

    kl_mc_all = []

    if gpu:
        target_all = target_all.cuda()
        prediction_all = prediction_all.cuda()
        
    for i in range(sequences_num):
        time_end = min(sequence_length, 10000)
        t_washout = int(0.25*sequence_length)
        time_steps = time_end - t_washout

        mu_gen= target_all[i, t_washout:time_end, :]
        mu_inf= prediction_all[i, t_washout:time_end, :]
        
        scaling = 1.
        cov_gen = tc.ones(dim).repeat(time_steps, 1)*scaling
        cov_inf = tc.ones(dim).repeat(time_steps, 1)*scaling

        kl_mc, _  = calculateKLMonteCarlo(mu_gen, cov_gen, mu_inf, cov_inf)
        kl_mc_all.append(kl_mc)

    kl_mc_all = tc.stack(kl_mc_all)
    kl_mc_avg = tc.mean(kl_mc_all)

    if gpu:
        target_all = target_all.cpu()
        prediction_all = prediction_all.cpu()
        kl_mc_avg = kl_mc_avg.cpu()
        
    target_all = target_all.numpy()
    prediction_all = prediction_all.numpy()
    kl_mc_avg = kl_mc_avg.numpy()
    
    return kl_mc_avg


def getFirstActiveIndex(parallel_group_interaction_length):
    if parallel_group_interaction_length > 0:
        return parallel_group_interaction_length
    else:
        return 0

def getLastActiveIndex(parallel_group_interaction_length):
    if parallel_group_interaction_length > 0:
        return -parallel_group_interaction_length
    else:
        return None

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))

    return gpu_memory_map


# UTILITIES FOR PARALLEL MODELS
def createParallelTrainingData(model):
    # GROUP NUMMER = WORKER ID
    group_num = model.parallel_group_num
    group_start = group_num * model.parallel_group_size  	# the start of the corresponded input dimension for the current model
    group_end = group_start + model.parallel_group_size		# the end ...
    pgil = model.parallel_group_interaction_length
    training_path_group = reformatParallelGroupDataPath(model, model.main_train_data_path, group_num, pgil)
    testing_path_group = reformatParallelGroupDataPath(model, model.main_test_data_path, group_num, pgil)
    if(not os.path.isfile(training_path_group)):
        if model.display_output == True: print("## Generating data for group {:d}-{:d} ##".format(group_start, group_end))
        with open(model.main_train_data_path, "rb") as file:
            data = pickle.load(file)
            train_sequence = data["train_input_sequence"][:, :model.RDIM]
            testing_ic_indexes = data["testing_ic_indexes"]
            dt = data["dt"]
            mle = data["mle"]
            del data
        train_sequence_group = createParallelGroupTrainingSequence(group_num, group_start, group_end, pgil, train_sequence)
        data = {"train_input_sequence":train_sequence_group, "testing_ic_indexes":testing_ic_indexes, "dt":dt, "mle":mle}
        with open(training_path_group, "wb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data
    # else:
    # 	print("Training data file already exist.")

    if(not os.path.isfile(testing_path_group)):
        if model.display_output == True: print("## Generating data for group {:d}-{:d} ##".format(group_start, group_end))
        with open(model.main_test_data_path, "rb") as file:
            data = pickle.load(file)
            test_sequence = data["test_input_sequence"][:, :model.RDIM]
            testing_ic_indexes = data["testing_ic_indexes"]
            dt = data["dt"]
            mle = data["mle"]
            del data
        test_sequence_group = createParallelGroupTrainingSequence(group_num, group_start, group_end, pgil, test_sequence)
        data = {"test_input_sequence":test_sequence_group, "testing_ic_indexes":testing_ic_indexes, "dt":dt, "mle":mle}
        # np.savetxt(testing_path_group[:-7]+".txt", test_sequence_group, fmt="%.6f", delimiter=",")
        with open(testing_path_group, "wb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data
    # else:
    # 	print("Testing data file already exist.")

    return training_path_group, testing_path_group

def createParallelGroupTrainingSequence(gn, gs, ge, ll, sequence):
    sequence = np.transpose(sequence)
    sequence_group = []
    sequence = Circ(sequence)
    print("Indexes considered {:d}-{:d}".format(gs-ll, ge+ll))
    for i in range(gs-ll, ge+ll):
        sequence_group.append(sequence[i])
    sequence_group = np.transpose(sequence_group)
    return sequence_group

def reformatParallelGroupDataPath(model, path, gn, ll):
    # Last 7 string objects are .pickle
    last = 7
    path_ = path[:-last] + "_G{:d}-from-{:d}_GS{:d}_GIL{:d}".format(gn, model.num_parallel_groups, model.parallel_group_size, ll) + path[-last:]
    return path_

class Circ(list):
    def __getitem__(self, idx):
        return super(Circ, self).__getitem__(idx % len(self))

def getModel(params_dict):
    sys.path.insert(0, global_params.py_models_path.format(params_dict["model_name"]))
    if params_dict["model_name"] == "standard_esn":
        import standard_esn as model
        return model.StandardESN(params_dict)
    elif params_dict["model_name"] == "pgclustered_esn":
        import pgclustered_esn as model
        return model.PGClusteredESN(params_dict)
    elif params_dict["model_name"] == "asym_pgclustered_esn":
        import asym_pgclustered_esn as model
        return model.AsymPGClusteredESN(params_dict)
    elif params_dict["model_name"] == "partially_pgclustered_esn":
        import partially_pgclustered_esn as model
        return model.PartPGClusteredESN(params_dict)
    elif params_dict["model_name"] == "moved_pgclustered_esn":
        import moved_pgclustered_esn as model
        return model.MovedPGClusteredESN(params_dict)
    elif params_dict["model_name"] == "randomly_clustered_esn":
        import randomly_clustered_esn as model
        return model.RandClusteredESN(params_dict)
    elif params_dict["model_name"] == "paralleled_esn":
        import paralleled_esn as model
        return model.ParalleledESN(params_dict)
    else:
        raise ValueError("model not found.")
    
def tuneModel(params_dict):
    hype_tuning_config_path = params_dict["hype_tuning_config_path"]
    report_path = params_dict["saving_path"] + "/Tuning/round{:}".format(params_dict["hyper_tuning_round_num"]) 
    searchBestHyperParameters(params_dict, hype_tuning_config_path, report_path)
    best_hyperparams, _ = readBestHypAfterTuning(hype_tuning_config_path, report_path)
    print(best_hyperparams)
    return 0

def tuneOneHyperParameter(params_dict):
    model = getModel(params_dict)
    model.hyperTuning()
    model.saveValidationResult()
    model.delete()
    del model
    return 0

def evaluateTunedModel(params_dict):
    hype_tuning_config_path = params_dict["hype_tuning_config_path"]
    report_path = params_dict["saving_path"] + "/Tuning/round{:}".format(params_dict["hyper_tuning_round_num"]) 
    # config = get_conf_from_json(hype_tuning_config_path)
    best_hyperparams, exp_name = readBestHypAfterTuning(hype_tuning_config_path, report_path)
    print(best_hyperparams)
    # params_dict.update(best_hyperparams)

    params_dict["radius"] = best_hyperparams["radius"]
    params_dict["sigma_input"] = best_hyperparams["sigma_input"]
    
    if params_dict["model_name"] == "paralleled_esn":
        trainParalleledModelMultipleTimes(params_dict.copy())
        testParalleledModelMultipleTimes(params_dict.copy())
    else:
        trainModelMultipleTimes(params_dict.copy())
        testModelMultipleTimes(params_dict.copy())
    averageResultOfModelMultipleTimes(params_dict.copy(), exp_name)
    return 0

def trainModel(params_dict):
    model = getModel(params_dict)
    model.train()
    model.saveModel()
    model.delete()
    del model
    return 0

def testModel(params_dict):
    model = getModel(params_dict)
    model.test()
    model.saveEvaluationResult()
    model.delete()
    del model
    return 0

def testModelAndSaveSequences(params_dict):
    model = getModel(params_dict)
    model.test()
    model.saveResult()
    model.delete()
    del model
    return 0

def plotModel(params_dict):
    model = getModel(params_dict)
    model.test()
    model.plotTestResult()
    model.delete()
    del model
    return 0

def plotSavedSequences(params_dict):
    model = getModel(params_dict)
    model.plotSavedResult()
    model.delete()
    del model
    return 0

def trainModelMultipleTimes(params_dict):
    # to be sure there is no bias in the results due to initialization.
    trial_num = params_dict["initialization_num"]
    for i in range(trial_num):
        model = getModel(params_dict)
        model.train()
        model.saveModel()
        model.delete()
        del model
        params_dict["worker_id"] += 1
    return 0

def trainParalleledModelMultipleTimes(params_dict):
    # to be sure there is no bias in the results due to initialization.
    trial_num = params_dict["initialization_num"]
    for i in range(trial_num):
        # Construct the command to execute the MPI parallel training
        train_command = [
            "mpiexec", "--oversubscribe", "-n", str(int(params_dict["RDIM"]/params_dict["parallel_group_size"])), "python3", "RUN.py", params_dict["model_name"],
            "--mode", "train",
            "--display_output", str(params_dict["display_output"]),
            "--worker_id", str(params_dict["worker_id"]),
            "--experiment_name", params_dict["experiment_name"],
            "--system_name", params_dict["system_name"],
            "--write_to_log", str(params_dict["write_to_log"]),
            "--N_train", str(params_dict["N_train"]),
            "--N_test", str(params_dict["N_test"]),
            "--RDIM", str(params_dict["RDIM"]),
            "--noise_level", str(params_dict["noise_level"]),
            "--scaler", params_dict["scaler"],
            "--approx_reservoir_size", str(params_dict["approx_reservoir_size"]),
            "--sparsity", str(params_dict["sparsity"]),
            "--p_in", str(params_dict["p_in"]),
            "--radius", str(params_dict["radius"]),
            "--sigma_input", str(params_dict["sigma_input"]),
            "--regularization", str(params_dict["regularization"]),
            "--dynamics_length", str(params_dict["dynamics_length"]),
            "--iterative_prediction_length", str(params_dict["iterative_prediction_length"]),
            "--num_test_ICS", str(params_dict["num_test_ICS"]),
            "--parallel_group_size", str(params_dict["parallel_group_size"]),
            "--parallel_group_interaction_length", str(params_dict["parallel_group_interaction_length"])
        ]
        try:
            subprocess.run(train_command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("MPI Error:", e.stderr)
        params_dict["worker_id"] += 1
    return 0

def testModelMultipleTimes(params_dict):
    trial_num = params_dict["initialization_num"]
    for i in range(trial_num):
        model = getModel(params_dict)
        model.test()
        model.saveEvaluationResult()
        model.delete()
        del model
        params_dict["worker_id"] += 1
    return 0

def testParalleledModelMultipleTimes(params_dict):
        # to be sure there is no bias in the results due to initialization.
    trial_num = params_dict["initialization_num"]
    # print("test started")
    for i in range(trial_num):
        # Construct the command to execute the MPI parallel training
        test_command = [
            "mpiexec", "--oversubscribe", "-n", str(int(params_dict["RDIM"]/params_dict["parallel_group_size"])), "python3", "RUN.py", params_dict["model_name"],
            "--mode", "test",
            "--display_output", str(params_dict["display_output"]),
            "--worker_id", str(params_dict["worker_id"]),
            "--experiment_name", params_dict["experiment_name"],
            "--system_name", params_dict["system_name"],
            "--write_to_log", str(params_dict["write_to_log"]),
            "--N_train", str(params_dict["N_train"]),
            "--N_test", str(params_dict["N_test"]),
            "--RDIM", str(params_dict["RDIM"]),
            "--noise_level", str(params_dict["noise_level"]),
            "--scaler", params_dict["scaler"],
            "--approx_reservoir_size", str(params_dict["approx_reservoir_size"]),
            "--sparsity", str(params_dict["sparsity"]),
            "--p_in", str(params_dict["p_in"]),
            "--radius", str(params_dict["radius"]),
            "--sigma_input", str(params_dict["sigma_input"]),
            "--regularization", str(params_dict["regularization"]),
            "--dynamics_length", str(params_dict["dynamics_length"]),
            "--iterative_prediction_length", str(params_dict["iterative_prediction_length"]),
            "--num_test_ICS", str(params_dict["num_test_ICS"]),
            "--parallel_group_size", str(params_dict["parallel_group_size"]),
            "--parallel_group_interaction_length", str(params_dict["parallel_group_interaction_length"])
        ]
        try:
            subprocess.run(test_command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("MPI Error:", e.stderr)
        params_dict["worker_id"] += 1
        # print("test one time finished")
    return 0

def plotParalleledModelResult(params_dict):
    plot_command = [
            "mpiexec", "--oversubscribe", "-n", str(int(params_dict["RDIM"]/params_dict["parallel_group_size"])), "python3", "RUN.py", params_dict["model_name"],
            "--mode", "plot",
            "--display_output", str(params_dict["display_output"]),
            "--worker_id", str(params_dict["worker_id"]),
            "--experiment_name", params_dict["experiment_name"],
            "--system_name", params_dict["system_name"],
            "--write_to_log", str(params_dict["write_to_log"]),
            "--N_train", str(params_dict["N_train"]),
            "--N_test", str(params_dict["N_test"]),
            "--RDIM", str(params_dict["RDIM"]),
            "--noise_level", str(params_dict["noise_level"]),
            "--scaler", params_dict["scaler"],
            "--approx_reservoir_size", str(params_dict["approx_reservoir_size"]),
            "--sparsity", str(params_dict["sparsity"]),
            "--p_in", str(params_dict["p_in"]),
            "--radius", str(params_dict["radius"]),
            "--sigma_input", str(params_dict["sigma_input"]),
            "--regularization", str(params_dict["regularization"]),
            "--dynamics_length", str(params_dict["dynamics_length"]),
            "--iterative_prediction_length", str(params_dict["iterative_prediction_length"]),
            "--num_test_ICS", str(params_dict["num_test_ICS"]),
            "--parallel_group_size", str(params_dict["parallel_group_size"]),
            "--parallel_group_interaction_length", str(params_dict["parallel_group_interaction_length"])
        ]
    try:
        subprocess.run(plot_command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("MPI Error:", e.stderr)
    return 0

def plotTunedModelResult(params_dict):
    hype_tuning_config_path = params_dict["hype_tuning_config_path"]
    report_path = params_dict["saving_path"] + "/Tuning/round{:}".format(params_dict["hyper_tuning_round_num"]) 
    best_hyperparams, _ = readBestHypAfterTuning(hype_tuning_config_path, report_path)
    print(best_hyperparams)
    # params_dict.update(best_hyperparams)
    params_dict["radius"] = best_hyperparams["radius"]
    params_dict["sigma_input"] = best_hyperparams["sigma_input"]
    if params_dict["model_name"] == "paralleled_esn":
        plotParalleledModelResult(params_dict)
        return 0
    model = getModel(params_dict)
    model.test()
    model.plotTestResult()
    model.delete()
    del model
    return 0

    
def averageResultOfModelMultipleTimes(params_dict, exp_name):
    # to be sure there is no bias in the results due to initialization.
    # to be sure there is no bias in the results due to initialization.
    rmnse_all_train = []
    num_accurate_pred_05_all_train = []
    num_accurate_pred_1_all_train = []
    error_freq_all_train = []
    d_temp_all_train = []
    d_geom_all_train = []
    
    rmnse_all_test = []
    num_accurate_pred_05_all_test = []
    num_accurate_pred_1_all_test = []
    error_freq_all_test = []
    d_temp_all_test = []
    d_geom_all_test = []

    trial_num = params_dict["initialization_num"]
    for i in range(trial_num):
        model = getModel(params_dict)
        model.loadEvaluationResult()

        rmnse_train = model.rmnse_avg_TRAIN
        num_accurate_pred_05_train = model.num_accurate_pred_05_avg_TRAIN
        num_accurate_pred_1_train = model.num_accurate_pred_1_avg_TRAIN 
        error_freq_train = model.error_freq_TRAIN
        d_temp_train = model.d_temp_TRAIN
        d_geom_train = model.d_geom_TRAIN

        rmnse_test = model.rmnse_avg_TEST 
        num_accurate_pred_05_test = model.num_accurate_pred_05_avg_TEST
        num_accurate_pred_1_test = model.num_accurate_pred_1_avg_TEST
        error_freq_test = model.error_freq_TEST
        d_temp_test = model.d_temp_TEST
        d_geom_test = model.d_geom_TEST

        model.delete()
        del model

        rmnse_all_train.append(rmnse_train)
        num_accurate_pred_05_all_train.append(num_accurate_pred_05_train)
        num_accurate_pred_1_all_train.append(num_accurate_pred_1_train)
        error_freq_all_train.append(error_freq_train)
        d_temp_all_train.append(d_temp_train)
        d_geom_all_train.append(d_geom_train)

        rmnse_all_test.append(rmnse_test)
        num_accurate_pred_05_all_test.append(num_accurate_pred_05_test)
        num_accurate_pred_1_all_test.append(num_accurate_pred_1_test)
        error_freq_all_test.append(error_freq_test)
        d_temp_all_test.append(d_temp_test)
        d_geom_all_test.append(d_geom_test)

        params_dict["worker_id"] += 1

    rmnse_all_train = np.array(rmnse_all_train)
    num_accurate_pred_05_all_train = np.array(num_accurate_pred_05_all_train)
    num_accurate_pred_1_all_train = np.array(num_accurate_pred_1_all_train)
    error_freq_all_train = np.array(error_freq_all_train)
    d_temp_all_train = np.array(d_temp_all_train)
    d_geom_all_train = np.array(d_geom_all_train)

    rmnse_all_test = np.array(rmnse_all_test)
    num_accurate_pred_05_all_test = np.array(num_accurate_pred_05_all_test)
    num_accurate_pred_1_all_test = np.array(num_accurate_pred_1_all_test)
    error_freq_all_test = np.array(error_freq_all_test)
    d_temp_all_test = np.array(d_temp_all_test)
    d_geom_all_test = np.array(d_geom_all_test)

    # Calculate means and stds
    rmnse_avg_train, rmnse_std_train = np.mean(rmnse_all_train), np.std(rmnse_all_train)
    num_accurate_pred_05_avg_train, num_accurate_pred_05_std_train = np.mean(num_accurate_pred_05_all_train), np.std(num_accurate_pred_05_all_train)
    num_accurate_pred_1_avg_train, num_accurate_pred_1_std_train = np.mean(num_accurate_pred_1_all_train), np.std(num_accurate_pred_1_all_train)
    error_freq_avg_train, error_freq_std_train = np.mean(error_freq_all_train), np.std(error_freq_all_train)
    d_temp_avg_train, d_temp_std_train = np.mean(d_temp_all_train), np.std(d_temp_all_train)
    d_geom_avg_train, d_geom_std_train = np.mean(d_geom_all_train), np.std(d_geom_all_train)

    rmnse_avg_test, rmnse_std_test = np.mean(rmnse_all_test), np.std(rmnse_all_test)
    num_accurate_pred_05_avg_test, num_accurate_pred_05_std_test = np.mean(num_accurate_pred_05_all_test), np.std(num_accurate_pred_05_all_test)
    num_accurate_pred_1_avg_test, num_accurate_pred_1_std_test = np.mean(num_accurate_pred_1_all_test), np.std(num_accurate_pred_1_all_test)
    error_freq_avg_test, error_freq_std_test = np.mean(error_freq_all_test), np.std(error_freq_all_test)
    d_temp_avg_test, d_temp_std_test = np.mean(d_temp_all_test), np.std(d_temp_all_test)
    d_geom_avg_test, d_geom_std_test = np.mean(d_geom_all_test), np.std(d_geom_all_test)

    # Print results with std
    print(f"AVERAGE RMNSE ERROR ON TRAIN DATA SET: {rmnse_avg_train} ± {rmnse_std_train}")
    print(f"AVG NUMBER OF ACCURATE 0.5 PREDICTIONS ON TRAIN DATA SET: {num_accurate_pred_05_avg_train} ± {num_accurate_pred_05_std_train}")
    print(f"AVG NUMBER OF ACCURATE 1 PREDICTIONS ON TRAIN DATA SET: {num_accurate_pred_1_avg_train} ± {num_accurate_pred_1_std_train}")
    print(f"POWER SPECTRUM MEAN ERROR ON TRAIN DATA SET: {error_freq_avg_train} ± {error_freq_std_train}")
    print(f"TEMPORAL DISTANCE ON TRAIN DATA SET: {d_temp_avg_train} ± {d_temp_std_train}")
    print(f"GEOMETRICAL DISTANCE ON TRAIN DATA SET: {d_geom_avg_train} ± {d_geom_std_train}")
    print("\n")

    print(f"AVERAGE RMNSE ERROR ON TEST DATA SET: {rmnse_avg_test} ± {rmnse_std_test}")
    print(f"AVG NUMBER OF ACCURATE 0.5 PREDICTIONS ON TEST DATA SET: {num_accurate_pred_05_avg_test} ± {num_accurate_pred_05_std_test}")
    print(f"AVG NUMBER OF ACCURATE 1 PREDICTIONS ON TEST DATA SET: {num_accurate_pred_1_avg_test} ± {num_accurate_pred_1_std_test}")
    print(f"POWER SPECTRUM MEAN ERROR ON TEST DATA SET: {error_freq_avg_test} ± {error_freq_std_test}")
    print(f"TEMPORAL DISTANCE ON TEST DATA SET: {d_temp_avg_test} ± {d_temp_std_test}")
    print(f"GEOMETRICAL DISTANCE ON TEST DATA SET: {d_geom_avg_test} ± {d_geom_std_test}")
   
    data = {}
    data["rmnse_avg_TRAIN"] = rmnse_avg_train
    data["num_accurate_pred_05_avg_TRAIN"] = num_accurate_pred_05_avg_train
    data["num_accurate_pred_1_avg_TRAIN"] = num_accurate_pred_1_avg_train
    data["error_freq_avg_TRAIN"] = error_freq_avg_train
    data["d_temp_avg_TRAIN"] = d_temp_avg_train
    data["d_geom_avg_TRAIN"] = d_geom_avg_train

    data["rmnse_std_TRAIN"] = rmnse_std_train
    data["num_accurate_pred_05_std_TRAIN"] = num_accurate_pred_05_std_train
    data["num_accurate_pred_1_std_TRAIN"] = num_accurate_pred_1_std_train
    data["error_freq_std_TRAIN"] = error_freq_std_train
    data["d_temp_std_TRAIN"] = d_temp_std_train
    data["d_geom_std_TRAIN"] = d_geom_std_train
    
    data["rmnse_avg_TEST"] = rmnse_avg_test
    data["num_accurate_pred_05_avg_TEST"] = num_accurate_pred_05_avg_test
    data["num_accurate_pred_1_avg_TEST"] = num_accurate_pred_1_avg_test
    data["error_freq_avg_TEST"] = error_freq_avg_test
    data["d_temp_avg_TEST"] = d_temp_avg_test
    data["d_geom_avg_TEST"] = d_geom_avg_test

    data["rmnse_std_TEST"] = rmnse_std_test
    data["num_accurate_pred_05_std_TEST"] = num_accurate_pred_05_std_test
    data["num_accurate_pred_1_std_TEST"] = num_accurate_pred_1_std_test
    data["error_freq_std_TEST"] = error_freq_std_test
    data["d_temp_std_TEST"] = d_temp_std_test
    data["d_geom_std_TEST"] = d_geom_std_test

    data_path = params_dict["saving_path"] + params_dict["results_dir"] + "round_{:}_{:}_system_name_{:}_reservoir_size_{:}_NTrain_{:}_Noise_{:}_Regularization_{:}".format(params_dict["hyper_tuning_round_num"], exp_name, \
                                                                                        params_dict["system_name"], params_dict["approx_reservoir_size"], params_dict["N_train"], params_dict["noise_level"], params_dict["regularization"]) + "_best_averaged_result.pickle"
    with open(data_path, "wb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        del data
    return 0


def getStandardESNParser(parser):
    parser.add_argument("--mode", help="train, test, train_test, tune, plot", type=str, required=True)
    parser.add_argument("--system_name", help="system_name", type=str, required=True)
    parser.add_argument("--experiment_name", help="experiment_name", type=str, required=True)
    parser.add_argument("--write_to_log", help="write_to_log", type=int, required=True)
    parser.add_argument("--N_train", help="data length for training", type=int, required=True)
    parser.add_argument("--N_test", help="data length  for test", type=int, required=True)
    parser.add_argument("--RDIM", help="RDIM", type=int, required=True)
    parser.add_argument("--approx_reservoir_size", help="approx_reservoir_size", type=int, required=True)
    parser.add_argument("--sparsity", help="sparsity", type=float, required=True)
    parser.add_argument("--p_in", help="p_in", type=float, required=True)
    parser.add_argument("--dynamics_length", help="dynamics_length", type=int, required=True)
    parser.add_argument("--iterative_prediction_length", help="iterative_prediction_length", type=int, required=True)
    parser.add_argument("--num_test_ICS", help="num_test_ICS", type=int, required=True)
    parser.add_argument("--scaler", help="scaler", type=str, required=True)

    parser.add_argument("--display_output", help="control the verbosity level of output , default True", type=int, required=False, default=1)
    parser.add_argument("--worker_id", help="work id", type=int, default=0, required=False)
    parser.add_argument("--radius", help="radius", type=float, default=0.1, required=False)
    parser.add_argument("--sigma_input", help="sigma_input", type=float, default=0.1, required=False)
    parser.add_argument("--regularization", help="regularization", type=float, default=0.01, required=False)
    parser.add_argument("--noise_level", help="noise level per mille in the training data", type=int, default=5, required=False)
    parser.add_argument("--initialization_num", help="number of times we try with different model initialization", type=int, default=5, required=False)

    # parser.add_argument("--hyper_tuning_method", help="the method for hyper tuning", type=str, default="random", required=False)
    parser.add_argument("--hyper_tuning_round_num", help="the round number of hyper tuning", type=int, default=0, required=False)
    parser.add_argument("--hyper_tuning_config_name", help="the name of the hyper tuning file", type=str, required=False)
    parser.add_argument("--loss", help="the metric used for hypertuning", type=str, default="d_temp", required=False)

    return parser

def getPGClusteredESNParser(parser):
    parser = getStandardESNParser(parser)
    parser.add_argument("--in_cluster_weight", help="control the connection strength inside and between the clusters in the reservoir", type=float, required=False, default=0.5)
    # parser.add_argument("--corresponding_input_weight", help="control the input strength of the corresponding input dimension for each cluster in the reservoir", type=float, required=False, default=0.8)
    parser.add_argument("--input_group_size", help="the size of dimensions of input for each cluster", type=int, required=False, default=1)
    parser.add_argument("--coupling_dims", help="Specify the dimensions to be coupled, accepts multiple integer values", type=int, 
                     nargs='+',  # '+' indicates one or more values
                     required=True  # if required = True, then all models need that parameter
                    )
    return parser

def getAsymPGClusteredESNParser(parser):
    parser = getStandardESNParser(parser)
    parser.add_argument("--in_cluster_weight", help="control the connection strength inside and between the clusters in the reservoir", type=float, required=False, default=0.5)
    # parser.add_argument("--corresponding_input_weight", help="control the input strength of the corresponding input dimension for each cluster in the reservoir", type=float, required=False, default=0.8)
    parser.add_argument("--coupling_dims", help="Specify the dimensions to be coupled for each cluster, accepts cluster # of list of integer values", type=str, 
                     required=True  # if required = True, then all models need that parameter
                    )
    parser.add_argument("--input_group_size", help="the size of dimensions of input for each cluster", type=int, required=False, default=1)
    return parser

def getPartPGClusteredESNParser(parser):
    parser = getStandardESNParser(parser)
    parser.add_argument("--in_cluster_weight", help="control the connection strength inside and between the clusters in the reservoir", type=float, required=False, default=0.5)
    # parser.add_argument("--corresponding_input_weight", help="control the input strength of the corresponding input dimension for each cluster in the reservoir", type=float, required=False, default=0.8)
    parser.add_argument("--input_group_size", help="the size of dimensions of input for each cluster", type=int, required=False, default=1)
    parser.add_argument("--coupling_dims", help="Specify the dimensions to be coupled, accepts multiple integer values", type=int, 
                     nargs='+',  # '+' indicates one or more values
                     required=True  # if required = True, then all models need that parameter
                    )
    parser.add_argument("--prob_wrong_couplings", help="probability of wrong couplings", type=float, required=True)
    return parser

def getMovedPGClusteredESNParser(parser):
    parser = getStandardESNParser(parser)
    parser.add_argument("--in_cluster_weight", help="control the connection strength inside and between the clusters in the reservoir", type=float, required=False, default=0.5)
    # parser.add_argument("--corresponding_input_weight", help="control the input strength of the corresponding input dimension for each cluster in the reservoir", type=float, required=False, default=0.8)
    parser.add_argument("--input_group_size", help="the size of dimensions of input for each cluster", type=int, required=False, default=1)
    parser.add_argument("--coupling_dims", help="Specify the dimensions to be coupled, accepts multiple integer values", type=int, 
                     nargs='+',  # '+' indicates one or more values
                     required=True  # if required = True, then all models need that parameter
                    )
    parser.add_argument("--move_dim", help="step of dimension for moving couplings", type=int, required=True)
    return parser

def getRandomlyClusteredESNParser(parser):
    parser = getStandardESNParser(parser)
    parser.add_argument("--input_group_size", help="the size of dimensions of input for each cluster", type=int, required=False, default=1)
    return parser

def getParalleledESNParser(parser):
    parser = getStandardESNParser(parser)
    parser.add_argument("--parallel_group_size", help="the size of dimensions of input for each reservoir. must be divisor of the input dimension", type=int, required=True)
    parser.add_argument("--parallel_group_interaction_length", help="The interaction length of each group. 0-rdim/2", type=int, required=True)
    return parser
