#!/bin/bash

cd ../../../Methods


# Default values
DEFAULT_MODE="train"
DEFAULT_SYSTEM_NAME="Lorenz96_F8_ALPHA1_DIM40"
DEFAULT_RESERVOIR_SIZE=1000
DEFAULT_SPARSITY=1
DEFAULT_PIN=1
DEFAULT_ROUND_NUM=0
DEFAULT_CONFIG_NAME="grid_search.json"
DEFAULT_NOISE_LEVEL=5
DEFAULT_REGULARIZATION=1e-2

# Read parameters
MODE=${1:-$DEFAULT_MODE}
SYSTEM_NAME=${2:-$DEFAULT_SYSTEM_NAME}
APPROX_RESERVOIR_SIZE=${3:-$DEFAULT_RESERVOIR_SIZE}
SPARSITY=${4:-$DEFAULT_SPARSITY}
PIN=${5:-$DEFAULT_PIN}
ROUND_NUM=${6:-$DEFAULT_ROUND_NUM}
CONFIG_NAME=${7:-$DEFAULT_CONFIG_NAME}
NOISE_LEVEL=${8:-$DEFAULT_NOISE_LEVEL}
REGULARIZATION=${9:-$DEFAULT_REGULARIZATION}


python3 RUN.py randomly_clustered_esn \
--mode $MODE \
--display_output 0 \
--system_name $SYSTEM_NAME \
--experiment_name Lorenz96_ANALYSIS \
--write_to_log 1 \
--N_train 100000 \
--N_test 100000 \
--RDIM 40 \
--scaler Standard \
--approx_reservoir_size $APPROX_RESERVOIR_SIZE \
--sparsity $SPARSITY \
--p_in $PIN \
--dynamics_length 2000 \
--iterative_prediction_length 7000 \
--num_test_ICS 10 \
--hyper_tuning_round_num $ROUND_NUM \
--hyper_tuning_config_name $CONFIG_NAME \
--loss d_temp \
--noise_level $NOISE_LEVEL \
--regularization $REGULARIZATION \