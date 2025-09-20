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
--parse_string "round_(\d+)_grid-search-(.+?)_reservoir_size_3000_NTrain_(\d+)_Noise_5_Regularization_(\d+\.\d+)_step_size_1" \
--x_value "Training data length" \
--x_value_group_num 3 \