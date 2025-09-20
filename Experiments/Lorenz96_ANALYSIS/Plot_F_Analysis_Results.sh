#!/bin/bash

cd ../../Methods

# Default values
DEFAULT_MODEL_TYPES=""
DEFAULT_CONFIG_NAME="grid_search.json"

# Read parameters
MODEL_TYPES=${1:-$DEFAULT_MODEL_TYPES}
CONFIG_NAME=${2:-$DEFAULT_CONFIG_NAME}

python3 ANALYSE.py plot_all_evaluation_results \
--mode plot_all_evaluation_results \
--model_types $MODEL_TYPES \
--experiment_name Lorenz96_ANALYSIS \
--parse_string "round_(\d+)_grid-search-for-F-(.+?)_system_name_Lorenz96_F(\d+)_ALPHA(\d+)_DIM40_reservoir_size_3000_NTrain_100000_Noise_5_Regularization_(\d+\.\d+)_step_size_1" \
--x_value F \
--x_value_group_num 3 \