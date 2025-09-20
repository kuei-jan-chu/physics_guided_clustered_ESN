import os

global_params = lambda:0
global_params.cluster = 'local'

if global_params.cluster == 'local':
    # print("## CONFIG: RUNNING IN LOCAL REPOSITORY.")
    config_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(os.path.dirname(config_path))

# print("PROJECT PATH={}".format(project_path))

global_params.global_utils_path = "./Utils"

global_params.saving_path = project_path + "/Results/{:s}"
global_params.project_path = project_path

global_params.training_data_path = project_path + "/Data/{:s}/Data/training_data_N{:}.pickle"
global_params.testing_data_path = project_path + "/Data/{:s}/Data/testing_data_N{:}.pickle"

# PATH TO LOAD THE PYTHON MODELS
global_params.py_models_path = "./Models/{:}"

# PATH TO LOAD THE FILES FOR HYPER TUNING EXPERIMENTS
global_params.exp_tuning_file_path = project_path + "/Experiments/{:s}/{:s}/hype_tuning_configs/{:s}"

# PATHS FOR SAVING RESULTS OF THE RUN
global_params.model_dir = "/Trained_Models/"
global_params.fig_dir = "/Figures/"
global_params.results_dir = "/Evaluation_Data/"
global_params.logfile_dir = "/Logfiles/"

