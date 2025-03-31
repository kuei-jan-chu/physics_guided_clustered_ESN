#!/bin/bash

cd ../../../Methods

# Default values
DEFAULT_MODEL_TYPES=""
DEFAULT_SPECIFIC_MODEL_TYPE=""
DEFAULT_SETTING_VALUE_NAME=""
DEFAULT_SETTING_VALUE_LIST=None

MODEL_TYPES=${1:-$DEFAULT_MODEL_TYPES}
SPECIFIC_MODEL_TYPE=${2:-$DEFAULT_SPECIFIC_MODEL_TYPE}
SETTING_VALUE_NAME=${3:-$DEFAULT_SETTING_VALUE_NAME}
SETTING_VALUE_LIST=${4:-$DEFAULT_SETTING_VALUE_LIST}



python3 ANALYSE.py plot_all_evaluation_results_with_one_model_have_multiple_settings \
--mode plot_all_evaluation_results_with_one_model_have_multiple_settings \
--model_types $MODEL_TYPES \
--experiment_name Lorenz96_F8_ALPHA1_DIM40 \
--parse_string "round_(\d+)_grid-search-(.+?)_reservoir_size_(\d+)_NTrain_100000_Noise_5_Regularization_(\d+\.\d+)" \
--x_value "Reservoir Size" \
--x_value_group_num 3 \
--specific_model_type $SPECIFIC_MODEL_TYPE \
--parse_string_for_specific_model  "round_(\d+)_grid-search-for_pwc_(.+?)-partially_pgclustered_esn_(.+?)_reservoir_size_(\d+)_NTrain_100000_Noise_5_Regularization_(\d+\.\d+)" \
--setting_value_name $SETTING_VALUE_NAME \
--setting_value_list_for_specific_model $SETTING_VALUE_LIST \
--setting_value_group_num_for_specific_model 2 \
--x_value_group_num_for_specific_model 4 \