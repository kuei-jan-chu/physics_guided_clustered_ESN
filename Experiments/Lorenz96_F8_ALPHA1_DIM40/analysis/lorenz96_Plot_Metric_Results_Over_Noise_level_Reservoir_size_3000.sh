#!/bin/bash

cd ../../../Methods

# Default values
DEFAULT_MODEL_TYPES=""

# Read parameters
MODEL_TYPES=${1:-$DEFAULT_MODEL_TYPES}

python3 ANALYSE.py plot_all_evaluation_results \
--mode plot_all_evaluation_results \
--model_types $MODEL_TYPES \
--experiment_name Lorenz96_F8_ALPHA1_DIM40 \
--parse_string "round_(\d+)_(.+?)_reservoir_size_3000_NTrain_100000_Noise_(\d+)_Regularization_(\d+\.\d+)" \
--x_value "Noise level" \
--x_value_group_num 3 \