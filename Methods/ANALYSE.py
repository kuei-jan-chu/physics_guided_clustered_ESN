import sys

from Config.global_conf import global_params
sys.path.insert(0, global_params.global_utils_path)
from analyse_utils import *
import argparse
    
def runModel(params_dict):
    if params_dict["mode"] in ["plot_all_evaluation_results"]:
        plotModelsResult(params_dict)
    if params_dict["mode"] in ["plot_all_evaluation_results_with_one_model_have_multiple_settings"]:
        plotModelsResultWithOneModelHaveMultipleSettings(params_dict)
    if params_dict["mode"] in ["plot_nrmse_over_all_model_with_size_fixed"]:
        plotNRMSE(params_dict)
    if params_dict["mode"] in ["plot_mle_analysis"]:
        plotMLE(params_dict)
    if params_dict["mode"] in ["plot_all_nonlinearity_and_icw_analysis_evaluation_results"]:
        plotNonlinearityAndIcwResults(params_dict)
    return 0


def defineParser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='your analyse target', dest='analyse_target')

    plot_all_evaluation_results_parser = subparsers.add_parser("plot_all_evaluation_results")
    plot_all_evaluation_results_parser = getEvaluationResultsParser(plot_all_evaluation_results_parser)

    plot_all_evaluation_results_with_one_model_have_multiple_settings_parser = subparsers.add_parser("plot_all_evaluation_results_with_one_model_have_multiple_settings")
    plot_all_evaluation_results_with_one_model_have_multiple_settings_parser = getEvaluationResultsWithOneModelHaveMultipleSettingsParser(plot_all_evaluation_results_with_one_model_have_multiple_settings_parser)

    plot_nrmse_of_all_model_with_size_fixed_parser = subparsers.add_parser("plot_nrmse_over_all_model_with_size_fixed")
    plot_nrmse_of_all_model_with_size_fixed_parser = getNRMSEofAllModelsWithSizeFixedParser(plot_nrmse_of_all_model_with_size_fixed_parser)

    plot_mle_analysis_parser = subparsers.add_parser("plot_mle_analysis")
    plot_mle_analysis_parser = getMLEAnalysisParser(plot_mle_analysis_parser)

    plot_all_nonlinearity_and_icw_analysis_evaluation_results_parser = subparsers.add_parser("plot_all_nonlinearity_and_icw_analysis_evaluation_results")
    plot_all_nonlinearity_and_icw_analysis_evaluation_results_parser = getNonlinearityAndIcwAnalysisParser(plot_all_nonlinearity_and_icw_analysis_evaluation_results_parser)
    
    return parser

def main():
    parser = defineParser()
    args = parser.parse_args()
    args_dict = args.__dict__


    # for key in args_dict:
    #     print(key)

    # DEFINE PATHS AND DIRECTORIES
    args_dict["saving_path"] = global_params.saving_path.format(args_dict["experiment_name"]) + "/analyse/{:}".format(args_dict["analyse_target"])
    args_dict["fig_dir"] = global_params.fig_dir
    args_dict["results_dir"] = global_params.results_dir

    runModel(args_dict)


if __name__ == '__main__':
    main()