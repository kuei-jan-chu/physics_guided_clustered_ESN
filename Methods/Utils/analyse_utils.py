import re
import sys
from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from global_utils import *
from plotting_utils import *

def _plotModelsMetricResults(params_dict, evaluation_metric):
    all_models_results = []
    for model in params_dict["model_types"]:
        evaluation_results_path = global_params.saving_path.format(params_dict["experiment_name"]) + "/{:}/Evaluation_Data".format(model) 
        save_file = path.join(evaluation_results_path, "round_")
        best_results_paths = glob(f"{save_file}*")

        model_results = []

        # print(best_results_paths)
        for result_path in best_results_paths:
            match = re.search(params_dict["parse_string"], result_path)
            # print(match)
            if match != None:
                x_value = float(match.group(params_dict["x_value_group_num"]))
                with open(result_path, "rb") as file:
                    data = pickle.load(file)
                    result = {
                        params_dict["x_value"]: x_value,
                        "metric_result_avg_train": data[evaluation_metric + "_avg_TRAIN"],
                        "metric_result_std_train": data[evaluation_metric + "_std_TRAIN"],
                        "metric_result_avg_test": data[evaluation_metric + "_avg_TEST"],
                        "metric_result_std_test": data[evaluation_metric + "_std_TEST"],
                    }
                model_results.append(result)
        
        # Sort the model_results by reservoir size (first element of each tuple)
        model_results.sort(key=lambda x: x[params_dict["x_value"]])
        # print(len(model_results))
        all_models_results.append(
            {
                "model_name": model,
                "model_results": model_results
            })
    file_path = params_dict["saving_path"] + "_" + params_dict["x_value"] + params_dict["results_dir"]
    os.makedirs(file_path, exist_ok=True)
    save_file = path.join(file_path, "evaluate_by_{:}_result.json".format(evaluation_metric))
    with open(save_file, "w+") as file:
        json.dump(all_models_results, file)

    fig_dir_path = params_dict["saving_path"] + "_"  + params_dict["x_value"] + params_dict["fig_dir"]
    os.makedirs(fig_dir_path, exist_ok=True)
    plotEvaluationResultOverAllXValues(all_models_results, params_dict, fig_dir_path, evaluation_metric)

def _plotModelsMetricResultsWithOneModelHaveMultipleSettings(params_dict, evaluation_metric):
    all_models_results = []
    for model in params_dict["model_types"]:
        evaluation_results_path = global_params.saving_path.format(params_dict["experiment_name"]) + "/{:}/Evaluation_Data".format(model) 
        save_file = path.join(evaluation_results_path, "round_")
        best_results_paths = glob(f"{save_file}*")

        model_results = []
        all_specific_model_results = []
        if model == params_dict["specific_model_type"]:
            for setting_value in params_dict["setting_value_list_for_specific_model"]:
                all_specific_model_results.append(
                    {
                    "model_name" : labelNameForModels(model)+ ":" + params_dict["setting_value_name"] + "=" + str(setting_value),
                    "model_results": []
                })

        # print(best_results_paths)
        for result_path in best_results_paths:
            if model == params_dict["specific_model_type"]:
                match = re.search(params_dict["parse_string_for_specific_model"], result_path)
            else:
                match = re.search(params_dict["parse_string"], result_path)
            # print(match)
            if match != None:
                if model == params_dict["specific_model_type"]:
                    x_value = float(match.group(params_dict["x_value_group_num_for_specific_model"]))
                else:
                    x_value = float(match.group(params_dict["x_value_group_num"]))
                with open(result_path, "rb") as file:
                    data = pickle.load(file)
                    result = {
                        params_dict["x_value"]: x_value,
                        "metric_result_avg_train": data[evaluation_metric + "_avg_TRAIN"],
                        "metric_result_std_train": data[evaluation_metric + "_std_TRAIN"],
                        "metric_result_avg_test": data[evaluation_metric + "_avg_TEST"],
                        "metric_result_std_test": data[evaluation_metric + "_std_TEST"],
                    }
                if model == params_dict["specific_model_type"]:
                    # print(result)
                    setting_value = float(match.group(params_dict["setting_value_group_num_for_specific_model"]))
                    if setting_value == int(setting_value):
                        setting_value = int(setting_value)
                    for entry in all_specific_model_results:
                        if entry["model_name"] == labelNameForModels(model)+ ":" + params_dict["setting_value_name"] + "=" + str(setting_value):    
                            entry["model_results"].append(result)
                            break 
                else:        
                    model_results.append(result)
        if model == params_dict["specific_model_type"]:
            # Sort results by the x_value for each ICW
            for entry in all_specific_model_results:
                entry["model_results"].sort(key=lambda x: x[params_dict["x_value"]])
            all_models_results.extend(all_specific_model_results)
        else:
            # Sort the model_results by reservoir size (first element of each tuple)
            model_results.sort(key=lambda x: x[params_dict["x_value"]])
            # print(len(model_results))
            all_models_results.append(
                {
                    "model_name": model,
                    "model_results": model_results
                })
            
    # print(all_models_results)
    file_path = params_dict["saving_path"] + "_" + params_dict["x_value"] + "/{:}".format(params_dict["setting_value_name"]) + params_dict["results_dir"]
    os.makedirs(file_path, exist_ok=True)
    save_file = path.join(file_path, "evaluate_by_{:}_result.json".format(evaluation_metric))
    with open(save_file, "w+") as file:
        json.dump(all_models_results, file)

    fig_dir_path = params_dict["saving_path"] + "_"  + params_dict["x_value"] + "/{:}".format(params_dict["setting_value_name"]) + params_dict["fig_dir"]
    os.makedirs(fig_dir_path, exist_ok=True)
    plotEvaluationResultOverAllXValues(all_models_results, params_dict, fig_dir_path, evaluation_metric)


def plotNRMSE(params_dict):
    models = []
    nrmses_test_for_all_models = []
    nrmses_train_for_all_models = []
    for i, model_type in enumerate(params_dict["model_types"]):
        hype_tuning_config_path = global_params.exp_tuning_file_path \
            .format(params_dict["experiment_name"], model_type, params_dict["hyper_tuning_config_name"])
        model_path = global_params.saving_path.format(params_dict["experiment_name"]) + "/{:}".format(model_type)
        report_path = model_path + "/Tuning/round{:}".format(params_dict["hyper_tuning_round_num"]) 
        best_hyperparams, exp_name = readBestHypAfterTuning(hype_tuning_config_path, report_path)
        params_dict_for_model = params_dict.copy()
        params_dict_for_model.update(best_hyperparams)
        params_dict_for_model["sparsity"] = params_dict["sparsity_list"][i]
        params_dict_for_model["p_in"] = params_dict["p_in_list"][i]
        params_dict_for_model["model_name"] = model_type
        params_dict_for_model["saving_path"] = global_params.saving_path.format(params_dict_for_model["experiment_name"]) + "/{:}".format(model_type)
        params_dict_for_model["model_dir"] = global_params.model_dir
        params_dict_for_model["fig_dir"] = global_params.fig_dir
        params_dict_for_model["results_dir"] = global_params.results_dir
        params_dict_for_model["logfile_dir"] = global_params.logfile_dir
        params_dict_for_model["train_data_path"] = global_params.training_data_path.format(params_dict_for_model["system_name"], params_dict_for_model["N_train"])
        params_dict_for_model["test_data_path"] = global_params.testing_data_path.format(params_dict_for_model["system_name"], params_dict_for_model["N_test"])
        
        model = getModel(params_dict_for_model)
        model.loadEvaluationResult()
        # model_result = model.loadResult()
        if i == 0:
            num_test_ICS = model.num_test_ICS
            testing_ic_indexes = model.testing_ic_indexes_TEST
            dt = model.dt_TEST
            data_mle = model.data_mle_TEST
        models.append(model_type)
        nrmses_test_for_all_models.append(model.rmnse_all_TEST)
        nrmses_train_for_all_models.append(model.rmnse_all_TRAIN)
    # fig_dir_path = params_dict["saving_path"] + params_dict["fig_dir"] + str(params_dict["approx_reservoir_size"]) + '/'+ params_dict["hyper_tuning_config_name"] + '/' + str(params_dict["hyper_tuning_round_num"])
    fig_dir_path = params_dict["saving_path"] + params_dict["fig_dir"] + str(params_dict["approx_reservoir_size"])
    os.makedirs(fig_dir_path, exist_ok=True)
    plotFirstThreeNRMSEForAllModels(models, nrmses_test_for_all_models, num_test_ICS, testing_ic_indexes, dt, data_mle, fig_dir_path, "TEST")
    plotFirstThreeNRMSEForAllModels(models, nrmses_train_for_all_models, num_test_ICS, testing_ic_indexes, dt, data_mle, fig_dir_path, "TRAIN")
    return 0

def plotModelsResult(params_dict):
    # evaluation_metric = params_dict["evaluation_metric"]
    for evaluation_metric in ["rmnse", "num_accurate_pred_05", "num_accurate_pred_1", "error_freq", "d_temp", "d_geom"]:
        _plotModelsMetricResults(params_dict, evaluation_metric)

def plotModelsResultWithOneModelHaveMultipleSettings(params_dict):
    for evaluation_metric in ["rmnse", "num_accurate_pred_05", "num_accurate_pred_1", "error_freq", "d_temp", "d_geom"]:
        _plotModelsMetricResultsWithOneModelHaveMultipleSettings(params_dict, evaluation_metric)


def plotMLE(params_dict):
    mle_list = []
    x_value_list = []
    for system, x_value in zip(params_dict["system_list"], params_dict["x_value_list"]):
        data_path = global_params.project_path + f"/Data/{system}/Simulation_Data"
        file_path = glob(path.join(data_path, "*data.pickle"))[0]
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            mle = data["MLE"] 
            mle_list.append(mle)
            x_value_list.append(x_value)

    file_path = params_dict["saving_path"] + "_" + params_dict["x_value"] + params_dict["fig_dir"]
    os.makedirs(file_path, exist_ok=True)
    file_name = "mle_analysis_over_{:}".format(params_dict["x_value"])
    plotTwoList(mle_list, x_value_list, "MLE", params_dict["x_value"], file_path, file_name)
    return 0

def _plotNonlinearityAndIcwResultForOneMetric(params_dict, evaluation_metric):
    evaluation_results_path = global_params.saving_path.format(params_dict["experiment_name"]) + "/{:}/Evaluation_Data".format(params_dict["model"]) 
    save_file = path.join(evaluation_results_path, "round_")
    best_results_paths = glob(f"{save_file}*")
    all_icw_results = []
    for icw in params_dict["icw_list"]:
        all_icw_results.append(
            {
            "model_name" : "icw" + str(icw),
            "model_results": []
        })

    for result_path in best_results_paths:
            match = re.search(params_dict["parse_string"], result_path)
            if match != None:
                icw_value = float(match.group(params_dict["icw_value_group_num"]))
                x_value = float(match.group(params_dict["x_value_group_num"]))
                with open(result_path, "rb") as file:
                    data = pickle.load(file)
                    result = {
                        params_dict["x_value"]: x_value,
                        "metric_result_avg_train": data[evaluation_metric + "_avg_TRAIN"],
                        "metric_result_std_train": data[evaluation_metric + "_std_TRAIN"],
                        "metric_result_avg_test": data[evaluation_metric + "_avg_TEST"],
                        "metric_result_std_test": data[evaluation_metric + "_std_TEST"],
                    }
                for entry in all_icw_results:
                    if entry["model_name"] == "icw" + str(icw_value):
                        entry["model_results"].append(result)
                        break  # Exit loop once found
    # Sort results by the x_value for each ICW
    for entry in all_icw_results:
        entry["model_results"].sort(key=lambda x: x[params_dict["x_value"]])
    fig_dir_path = params_dict["saving_path"] + "_"  + params_dict["x_value"] + params_dict["fig_dir"] + '/' + str(params_dict["approx_reservoir_size"]) + '/'
    os.makedirs(fig_dir_path, exist_ok=True)
    plotEvaluationResultOverAllXValues(all_icw_results, params_dict, fig_dir_path, evaluation_metric)

def plotNonlinearityAndIcwResults(params_dict):
    # evaluation_metric = params_dict["evaluation_metric"]
    for evaluation_metric in ["rmnse", "num_accurate_pred_05", "num_accurate_pred_1", "error_freq", "d_temp", "d_geom"]:
        _plotNonlinearityAndIcwResultForOneMetric(params_dict, evaluation_metric)

def getEvaluationResultsParser(parser):
    parser.add_argument("--mode", help="plot_all_evaluation_results", type=str, required=True)
    parser.add_argument("--model_types", help="all the evaluated models", type=str, nargs='+', required=True)
    parser.add_argument("--experiment_name", help="experiment_name", type=str, required=True)
    parser.add_argument("--parse_string", help="the parse string of the result file", type=str, required=True)
    parser.add_argument("--x_value", help="the value changed on the x axis", type=str, required=True)
    parser.add_argument("--x_value_group_num", help="the group num for the value changed on the x axis", type=int, required=True)
    return parser

def getEvaluationResultsWithOneModelHaveMultipleSettingsParser(parser):
    parser.add_argument("--mode", help="plot_all_evaluation_results_with_one_model_have_multiple_settings", type=str, required=True)
    parser.add_argument("--model_types", help="all the evaluated models", type=str, nargs='+', required=True)
    parser.add_argument("--experiment_name", help="experiment_name", type=str, required=True)
    parser.add_argument("--parse_string", help="the parse string of the result file", type=str, required=True)
    parser.add_argument("--x_value", help="the value changed on the x axis", type=str, required=True)
    parser.add_argument("--x_value_group_num", help="the group num for the value changed on the x axis", type=int, required=True)
    parser.add_argument("--specific_model_type", help="the model with multiple settings", type=str, required=True)
    parser.add_argument("--setting_value_name", help="the setting value name for the specific model", type=str, required=True)
    parser.add_argument("--setting_value_list_for_specific_model", help="the setting value list for the specific model", nargs='+', required=True)
    parser.add_argument("--parse_string_for_specific_model", help="the parse string of the result file for the specific model", type=str, required=True)
    parser.add_argument("--setting_value_group_num_for_specific_model", help="the group num for the specific value of the specific model", type=int, required=True)
    parser.add_argument("--x_value_group_num_for_specific_model", help="the group num for the value changed on the x axis for the specific model", type=int, required=True)

def getNRMSEofAllModelsWithSizeFixedParser(parser):
    parser.add_argument("--mode", help="plot_nrmse_over_all_model_with_size_fixed", type=str, required=True)
    parser.add_argument("--model_types", help="all the evaluated models", type=str, nargs='+', required=True)
    parser.add_argument("--experiment_name", help="experiment_name", type=str, required=True)
    parser.add_argument("--system_name", help="system_name", type=str, required=True)
    parser.add_argument("--hyper_tuning_config_name", help="the name of the hyper tuning file", type=str, required=True)
    parser.add_argument("--hyper_tuning_round_num", help="the round number of hyper tuning", type=int, default=0, required=False)
    parser.add_argument("--N_train", help="data length for training", type=int, required=True)
    parser.add_argument("--N_test", help="data length  for test", type=int, required=True)
    parser.add_argument("--RDIM", help="RDIM", type=int, required=True)
    parser.add_argument("--approx_reservoir_size", help="approx_reservoir_size", type=int, required=True)
    parser.add_argument("--sparsity_list", help="sparsity", type=float, nargs='+', required=True)
    parser.add_argument("--p_in_list", help="p_in", type=float, nargs='+', required=True)
    parser.add_argument("--dynamics_length", help="dynamics_length", type=int, required=True)
    parser.add_argument("--iterative_prediction_length", help="iterative_prediction_length", type=int, required=True)
    parser.add_argument("--num_test_ICS", help="num_test_ICS", type=int, required=True)
    parser.add_argument("--scaler", help="scaler", type=str, required=True)
    parser.add_argument("--worker_id", help="work id", type=int, default=0, required=False)
    parser.add_argument("--radius", help="radius", type=float, default=0.1, required=False)
    parser.add_argument("--sigma_input", help="sigma_input", type=float, default=0.1, required=False)
    parser.add_argument("--regularization", help="regularization", type=float, default=0.01, required=False)
    parser.add_argument("--noise_level", help="noise level per mille in the training data", type=int, default=5, required=False)
    parser.add_argument("--input_group_size", help="the size of dimensions of input for each cluster", type=int, required=False, default=1)
    parser.add_argument("--in_cluster_weight", help="control the connection strength inside and between the clusters in the reservoir", type=float, required=False, default=0.5)
    parser.add_argument("--display_output", help="control the verbosity level of output , default True", type=int, required=False, default=0)
    parser.add_argument("--write_to_log", help="write_to_log", type=int, required=False, default=1)
    parser.add_argument("--coupling_dims", help="Specify the dimensions to be coupled, accepts multiple integer values", type=int, 
                     nargs='+',  # '+' indicates one or more values
                     required=True  # if required = True, then all models need that parameter
                    )
    parser.add_argument("--parallel_group_size", help="the size of dimensions of input for each reservoir. must be divisor of the input dimension", type=int, required=True)
    parser.add_argument("--parallel_group_interaction_length", help="The interaction length of each group. 0-rdim/2", type=int, required=True)
    return parser


def getMLEAnalysisParser(parser):
    parser.add_argument("--mode", help="plot_mle_analysis", type=str, required=True)
    parser.add_argument("--experiment_name", help="experiment_name", type=str, required=True)
    parser.add_argument("--system_list", help="the list of system for plotting the mle", type=str, nargs='+', required=True)
    parser.add_argument("--x_value", help="the value changed on the x axis", type=str, required=False)
    parser.add_argument("--x_value_list", help="the values on the x axis", type=int, nargs='+', required=True)
    return parser


def getNonlinearityAndIcwAnalysisParser(parser):
    parser.add_argument("--mode", help="plot_all_nonlinearity_and_icw_analysis_evaluation_results", type=str, required=True)
    parser.add_argument("--model", help="the evaluated model", type=str, required=True)
    parser.add_argument("--icw_list", help="the in cluster weight list", type=float, nargs='+', required=True)
    parser.add_argument("--experiment_name", help="experiment_name", type=str, required=True)
    parser.add_argument("--parse_string", help="the parse string for the result file", type=str, required=False)
    parser.add_argument("--x_value", help="the value changed on the x axis", type=str, required=False)
    parser.add_argument("--x_value_group_num", help="the group num for the value changed on the x axis", type=int, required=False)
    parser.add_argument("--icw_value_group_num", help="the group num for the icw value ", type=int, required=False)
    parser.add_argument("--approx_reservoir_size", help="approx_reservoir_size", type=int, required=True)
    return parser