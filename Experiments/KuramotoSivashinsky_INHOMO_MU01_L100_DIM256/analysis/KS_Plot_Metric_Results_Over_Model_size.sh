#!/bin/bash

cd ../../../Methods

# Default values
DEFAULT_MODEL_TYPES=""

# Read parameters
MODEL_TYPES=${1:-$DEFAULT_MODEL_TYPES}

python3 ANALYSE.py plot_all_evaluation_results \
--mode plot_all_evaluation_results \
--model_types $MODEL_TYPES \
--experiment_name KuramotoSivashinsky_INHOMO_MU01_L100_DIM256 \
--parse_string "round_(\d+)_(.+?)_reservoir_size_(\d+)_NTrain_100000_Noise_5_Regularization_(\d+\.\d+)_step_size_1" \
--x_value "Reservoir size" \
--x_value_group_num 3 \