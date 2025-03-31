import sys
from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from global_utils import *

import argparse

def runModel(params_dict):
    if params_dict["mode"] in ["tune"]:
        tuneModel(params_dict)
    if params_dict["mode"] in ["tune_one_hyper_parameter"]:
        tuneOneHyperParameter(params_dict)
    if params_dict["mode"] in ["evaluate_tuned_model"]:
        evaluateTunedModel(params_dict)
    if params_dict["mode"] in ["train", "train_and_test"]:
        trainModel(params_dict)
    if params_dict["mode"] in ["test", "train_and_test"]:
        testModel(params_dict)
    if params_dict["mode"] in ["test_and_save_for_plot"]:
        testModelAndSaveSequences(params_dict)
    if params_dict["mode"] in ["train_multiple_times", "train_and_test_multiple_times"]:
        trainModelMultipleTimes(params_dict)
    if params_dict["mode"] in ["test_multiple_times", "train_and_test_multiple_times"]:
        testModelMultipleTimes(params_dict)
    if params_dict["mode"] in ["average_results"]:
        averageResultOfModelMultipleTimes(params_dict)
    if params_dict["mode"] in ["plot_tuned_model"]:
        plotTunedModelResult(params_dict)
    if params_dict["mode"] in ["plot"]:
        plotModel(params_dict)
    if params_dict["mode"] in ["plot_saved_sequences"]:
        plotSavedSequences(params_dict)
    return 0


def defineParser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Selection of the model.', dest='model_name')

    standard_esn_parser = subparsers.add_parser("standard_esn")
    standard_esn_parser = getStandardESNParser(standard_esn_parser)

    pgclustered_esn_parser = subparsers.add_parser("pgclustered_esn")
    pgclustered_esn_parser = getPGClusteredESNParser(pgclustered_esn_parser)

    asym_pgclustered_esn_parser = subparsers.add_parser("asym_pgclustered_esn")
    asym_pgclustered_esn_parser = getAsymPGClusteredESNParser(asym_pgclustered_esn_parser)

    partially_pgclustered_esn_parser = subparsers.add_parser("partially_pgclustered_esn")
    partially_pgclustered_esn_parser = getPartPGClusteredESNParser(partially_pgclustered_esn_parser)

    moved_pgclustered_esn_parser = subparsers.add_parser("moved_pgclustered_esn")
    moved_pgclustered_esn_parser = getMovedPGClusteredESNParser(moved_pgclustered_esn_parser)

    randomly_clustered_esn_parser = subparsers.add_parser("randomly_clustered_esn")
    randomly_clustered_esn_parser = getRandomlyClusteredESNParser(randomly_clustered_esn_parser)

    paralleled_esn_parser = subparsers.add_parser("paralleled_esn")
    paralleled_esn_parser = getParalleledESNParser(paralleled_esn_parser)
    
    return parser

def main():
    parser = defineParser()
    args = parser.parse_args()
    # print(args.model_name)
    args_dict = args.__dict__

    # for key in args_dict:
    #     print(key)

    # DEFINE PATHS AND DIRECTORIES
    args_dict["saving_path"] = global_params.saving_path.format(args_dict["experiment_name"]) + "/{:}".format(args_dict["model_name"])
    args_dict["model_dir"] = global_params.model_dir
    args_dict["fig_dir"] = global_params.fig_dir
    args_dict["results_dir"] = global_params.results_dir
    args_dict["logfile_dir"] = global_params.logfile_dir
    args_dict["train_data_path"] = global_params.training_data_path.format(args_dict["system_name"], args_dict["N_train"])
    args_dict["test_data_path"] = global_params.testing_data_path.format(args_dict["system_name"], args_dict["N_test"])
    if args_dict["hyper_tuning_config_name"] != None:
        args_dict["hype_tuning_config_path"] = global_params.exp_tuning_file_path \
            .format(args_dict["experiment_name"], args_dict["model_name"], args_dict["hyper_tuning_config_name"])
    
    runModel(args_dict)

if __name__ == '__main__':
    main()

