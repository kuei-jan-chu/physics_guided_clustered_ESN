#!/bin/bash

cd ../../../Methods

# Default values
DEFAULT_MODE="train"
DEFAULT_RESERVOIR_SIZE=1000
DEFAULT_ROUND_NUM=0
DEFAULT_CONFIG_NAME="grid_search.json"
DEFAULT_NOISE_LEVEL=5
DEFAULT_REGULARIZATION=1e-2

# Read parameters
MODE=${1:-$DEFAULT_MODE}
APPROX_RESERVOIR_SIZE=${2:-$DEFAULT_RESERVOIR_SIZE}
ROUND_NUM=${3:-$DEFAULT_ROUND_NUM}
CONFIG_NAME=${4:-$DEFAULT_CONFIG_NAME}
NOISE_LEVEL=${5:-$DEFAULT_NOISE_LEVEL}
REGULARIZATION=${6:-$DEFAULT_REGULARIZATION}

python3 RUN.py randomly_clustered_esn \
--mode $MODE \
--display_output 0 \
--system_name KuramotoSivashinsky_INHOMO_MU01_L100_DIM256 \
--experiment_name  KuramotoSivashinsky_INHOMO_MU01_L100_DIM256  \
--write_to_log 1 \
--N_train 100000 \
--N_test 100000 \
--RDIM 256 \
--scaler Standard \
--approx_reservoir_size $APPROX_RESERVOIR_SIZE \
--sparsity 0.0938 \
--p_in 0.0938 \
--dynamics_length 2000 \
--iterative_prediction_length 7000 \
--num_test_ICS 10 \
--hyper_tuning_round_num $ROUND_NUM \
--hyper_tuning_config_name $CONFIG_NAME \
--loss d_temp \
--worker_id 0 \
--input_group_size 8 \
--noise_level $NOISE_LEVEL \
--regularization $REGULARIZATION \