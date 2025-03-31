#!/bin/bash

cd ../../../Methods

# Default values
DEFAULT_MODE="train"
DEFAULT_MODEL_TYPES=""
DEFAULT_CONFIG_NAME="grid_search.json"
DEFAULT_ROUND_NUM=0
DEFAULT_RESERVOIR_SIZE=1000

# Read parameters
MODE=${1:-$DEFAULT_MODE}
MODEL_TYPES=${2:-$DEFAULT_MODEL_TYPES}
CONFIG_NAME=${3:-$DEFAULT_CONFIG_NAME}
ROUND_NUM=${4:-$DEFAULT_ROUND_NUM}
APPROX_RESERVOIR_SIZE=${5:-$DEFAULT_RESERVOIR_SIZE}

python3 ANALYSE.py plot_nrmse_over_all_model_with_size_fixed \
--mode $MODE \
--model_types $MODEL_TYPES \
--system_name Lorenz96_F8_ALPHA1_DIM40  \
--experiment_name Lorenz96_F8_ALPHA1_DIM40 \
--hyper_tuning_config_name $CONFIG_NAME \
--hyper_tuning_round_num $ROUND_NUM \
--N_train 100000 \
--N_test 100000 \
--RDIM 40 \
--scaler Standard \
--approx_reservoir_size $APPROX_RESERVOIR_SIZE \
--sparsity_list 1 0.1 0.1 1 \
--p_in_list 1 0.1 0.1 1 \
--dynamics_length 2000 \
--iterative_prediction_length 7000 \
--num_test_ICS 10 \
--worker_id 0 \
--noise_level 5 \
--regularization 1e-2 \
--in_cluster_weight 0.5 \
--coupling_dims -2 -1 1 \
--input_group_size 1 \
--parallel_group_size 1 \
--parallel_group_interaction_length 2 \