import json
from math import e
from os import path
import os
import pickle
import random
import time
import warnings
import numpy as np
from glob import glob
import sys
from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from itertools import product
from tqdm import tqdm  # Import tqdm for progress bars
import subprocess


def _getModel(params_dict):
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
    
def _getLossMetric(model, metric):
    if metric == "error_freq":
        return model.error_freq_TEST
    elif metric == "d_geom":
        return model.d_geom_TEST
    elif metric == "d_temp":
        return model.d_temp_TEST
    elif metric == "num_accurate_pred_05_avg":
        return - model.num_accurate_pred_05_avg_TEST
    elif metric == 'num_accurate_pred_1_avg':
        return - model.num_accurate_pred_1_avg_TEST
    else: 
        return e

    
def get_report_path(exp_name, base_path=None):
    base_path = "." if base_path is None else base_path
    report_path = path.join(base_path, exp_name)
    os.makedirs(report_path, exist_ok=True)
    return report_path

def get_conf_from_json(confpath):
    if not (path.isfile(confpath)):
        raise FileNotFoundError(f"Training conf '{confpath}' not found. ")
    else:
        config = {}
        with open(confpath, "r") as f:
            config = json.load(f)
        return config
        

def convert_hyperopt_indices_to_values(best, search_space):
    converted_best = {}
    for key in best:
        current_hyp_param = search_space[key]
        # Check if it's an Apply object (i.e., a hyperopt symbolic expression) with method switch
        if isinstance(current_hyp_param, pyll.Apply) and current_hyp_param.name == 'switch':
            # Extract the list of choices
            # print(current_hyp_param.pos_args)
            choices_list = current_hyp_param.pos_args[1:] # Skip the first element (this is the index)
            # Map the index of best[key] back to the actual value from the choices
            converted_best[key] = choices_list[best[key]].obj   # Retrieve the actual value from the hyperopt symbolic expression
        else:
            # For other hyperparameters, just copy them directly
            converted_best[key] = best[key]

    # print(converted_best)
    return converted_best

# Objective functions accepted by ReservoirPy must respect some conventions:
#  - the function must return a dict with at least a 'loss' key containing the result
# of the loss function. You can add any additional metrics or information with other 
# keys in the dict. See hyperopt documentation for more informations.
def _objective(params_dict, hype_tuning_config, hyper_args):    
    if params_dict["model_name"] == "paralleled_esn":
        return _objectiveParalleledModel(params_dict, hype_tuning_config, hyper_args)
    # You can access anything you put in the config file from the 'hype_tuning_config' parameter.
    # the times we repeat for model using each hyperparameter combination
    instances = hype_tuning_config["instances_per_trial"]
    
    # each hyperparameter we try with the first worker_id as 0, and + 1 for each instance in the loop
    # The seed should be changed across the instances, 
    # to be sure there is no bias in the results due to initialization.
    params_dict["worker_id"] = 0

    # get hyperparameter being searched
    params_dict.update(hyper_args)
    
    loss_all = []
    rmse_all = []
    rmnse_all = []
    num_accurate_pred_05_all = []
    num_accurate_pred_1_all = []
    error_freq_all = []
    d_geom_all = []
    d_temp_all = []
    for _ in range(instances):

        # define model by all the parameters
        model = _getModel(params_dict)
    
        # hyperTuning your model.
        model.hyperTuning()
        
        metric = params_dict["loss"]
        loss = _getLossMetric(model, metric)
        rmse = model.rmse_avg_TEST
        rmnse = model.rmnse_avg_TEST
        num_accurate_pred_05 = model.num_accurate_pred_05_avg_TEST
        num_accurate_pred_1 = model.num_accurate_pred_1_avg_TEST
        error_freq = model.error_freq_TEST
        d_geom = model.d_geom_TEST
        d_temp = model.d_temp_TEST
        
        del model
        
        loss_all.append(loss)
        rmse_all.append(rmse)
        rmnse_all.append(rmnse)
        num_accurate_pred_05_all.append(num_accurate_pred_05)
        num_accurate_pred_1_all.append(num_accurate_pred_1)
        error_freq_all.append(error_freq)
        d_geom_all.append(d_geom)
        d_temp_all.append(d_temp)

        # Change the seed between instances
        params_dict["worker_id"] += 1

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': np.mean(loss_all),
            'rmse_avg': np.mean(rmse_all),
            'rmnse_avg': np.mean(rmnse_all),            
            'num_accurate_pred_05_avg': np.mean(num_accurate_pred_05_all),
            'num_accurate_pred_1_avg': np.mean(num_accurate_pred_1_all),
            'error_freq': np.mean(error_freq_all),
            'd_geom': np.mean(d_geom_all),
            'd_temp': np.mean(d_temp_all)
            }

def  _objectiveParalleledModel(params_dict, hype_tuning_config, hyper_args):
    instances = hype_tuning_config["instances_per_trial"]
    params_dict["worker_id"] = 0
    params_dict.update(hyper_args)
    
    loss_all = []
    rmse_all = []
    rmnse_all = []
    num_accurate_pred_05_all = []
    num_accurate_pred_1_all = []
    error_freq_all = []
    d_geom_all = []
    d_temp_all = []

    for _ in range(instances):
        # Construct the command to execute the MPI parallel training
        tune_command = [
            "mpiexec", "--oversubscribe", "-n", str(int(params_dict["RDIM"]/params_dict["parallel_group_size"])), "python3", "RUN.py", params_dict["model_name"],
            "--mode", "tune_one_hyper_parameter",
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
            subprocess.run(tune_command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("MPI Error:", e.stderr)
        model = _getModel(params_dict)
        model.loadValidationResult()
        metric = params_dict["loss"]
        loss = _getLossMetric(model, metric)
        rmse = model.rmse_avg_TEST
        rmnse = model.rmnse_avg_TEST
        num_accurate_pred_05 = model.num_accurate_pred_05_avg_TEST
        num_accurate_pred_1 = model.num_accurate_pred_1_avg_TEST
        error_freq = model.error_freq_TEST
        d_geom = model.d_geom_TEST
        d_temp = model.d_temp_TEST
        
        del model
        
        loss_all.append(loss)
        rmse_all.append(rmse)
        rmnse_all.append(rmnse)
        num_accurate_pred_05_all.append(num_accurate_pred_05)
        num_accurate_pred_1_all.append(num_accurate_pred_1)
        error_freq_all.append(error_freq)
        d_geom_all.append(d_geom)
        d_temp_all.append(d_temp)

        # Change the seed between instances
        params_dict["worker_id"] += 1

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    tuning_result = {'loss': np.mean(loss_all),
            'rmse_avg': np.mean(rmse_all),
            'rmnse_avg': np.mean(rmnse_all),            
            'num_accurate_pred_05_avg': np.mean(num_accurate_pred_05_all),
            'num_accurate_pred_1_avg': np.mean(num_accurate_pred_1_all),
            'error_freq': np.mean(error_freq_all),
            'd_geom': np.mean(d_geom_all),
            'd_temp': np.mean(d_temp_all)
            }
    return tuning_result

def gridSearch(params_dict, config, report_path):
    # Extract hyperparameter search space
    search_space = config["hp_space"]
    # Create lists of all possible values for each hyperparameter
    hyperparams = {
        key: value  for key, value in search_space.items()
    }
    # Generate all combinations of hyperparameters using product
    hyperparam_combinations = list(product(*hyperparams.values()))
    hyperparam_keys = list(hyperparams.keys())
    best_hyperparams = None
    best_loss = float("inf")
    results = []

    # Add tqdm progress bar for the hyperparameter combinations loop
    with tqdm(total=len(hyperparam_combinations), desc="Grid Search Progress", unit="trial", ncols=100) as pbar:
        for hyperparam_values in hyperparam_combinations:
            # Create the hyperparameter dictionary for this combination
            hyper_args = dict(zip(hyperparam_keys, hyperparam_values))
            
            # Evaluate the combination
            try:
                start = time.time()
                returned_dict = _objective(params_dict, config, hyper_args)
                end = time.time()
                duration = end - start

                returned_dict["start_time"] = start
                returned_dict["duration"] = duration
                returned_dict["hyperparameters"] = hyper_args
                save_file = f"{returned_dict['loss']:.7f}_hyperopt_results"
                # Track the best combination
                if returned_dict["loss"] < best_loss:
                    best_loss = returned_dict["loss"]
                    best_hyperparams = hyper_args   
                # Save results
                results.append(returned_dict)
                
            except Exception as e:
                start = time.time()
                returned_dict = {
                    "status": "FAILED",
                    "start_time": start,
                    "error": str(e),
                }
                save_file = f"ERR{start}_grid_search_results"
                warnings.warn(f"Error during evaluation of {hyper_args}: {str(e)}")
                continue
            
            try:
                json_dict = {"returned_dict": returned_dict, "current_params": hyper_args}
                save_file = path.join(report_path, save_file)
                nb_save_file_with_same_loss = len(glob(f"{save_file}*"))
                save_file = f"{save_file}_{nb_save_file_with_same_loss+1}call.json"
                with open(save_file, "w+") as f:
                    json.dump(json_dict, f)
            except Exception as e:
                warnings.warn(
                    "Results of current simulation were NOT saved "
                    "correctly to JSON file." + 
                    str(e)
                )
            # Update progress bar and postfix
            current_best_postfix = {
                "Current Best Loss": f"{best_loss:.5f}",
            }
            pbar.set_postfix(current_best_postfix)
            pbar.update(1)

    if best_hyperparams == None:
        best_hyperparams = dict(zip(hyperparam_keys, random.choice(hyperparam_combinations)))
        warnings.warn("All hyperparameter trials failed or returned infinite loss. Randomly selecting a hyperparameter set.")
    # Save results to a file
    tuning_result = {
        "best": best_hyperparams,
        "results": results,
    }
    with open(report_path + "/tuning_result.pickle", "wb") as file:
        pickle.dump(tuning_result, file, pickle.HIGHEST_PROTOCOL)
    
    return best_hyperparams, results


def searchBestHyperParameters(params_dict, config_path, report_path=None):
    # Load configuration
    config = get_conf_from_json(config_path)
    report_path = get_report_path(config["exp_name"], report_path)
    gridSearch(params_dict, config, report_path)


def readBestHypAfterTuning(config_path, report_path=None):
    config = get_conf_from_json(config_path)
    report_path = get_report_path(config["exp_name"], report_path)
    with open(report_path + "/tuning_result.pickle", "rb") as file:
        data = pickle.load(file)
        best_hyperparams = data["best"]
        del data
    return best_hyperparams, config["exp_name"]