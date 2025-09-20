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
--parse_string "round_(\d+)_grid-search-(.+?)_reservoir_size_3000_NTrain_100000_Noise_5_Regularization_0.01_step_size_(\d+)" \
--x_value "Step size" \
--x_value_group_num 3 \