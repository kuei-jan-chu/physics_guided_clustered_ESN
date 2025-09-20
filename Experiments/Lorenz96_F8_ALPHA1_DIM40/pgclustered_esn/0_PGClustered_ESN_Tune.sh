#!/bin/bash

cd ../../../Methods

# Default values
DEFAULT_MODE="train"
DEFAULT_RESERVOIR_SIZE=1000
DEFAULT_SPARSITY=1
DEFAULT_PIN=1
DEFAULT_ROUND_NUM=0
DEFAULT_CONFIG_NAME="grid_search.json"
DEFAULT_NOISE_LEVEL=5
DEFAULT_REGULARIZATION=1e-2
DEFAULT_COUPLING_DIMS="-2 -1 1"

# Read parameters
MODE=${1:-$DEFAULT_MODE}
APPROX_RESERVOIR_SIZE=${2:-$DEFAULT_RESERVOIR_SIZE}
SPARSITY=${3:-$DEFAULT_SPARSITY}
PIN=${4:-$DEFAULT_PIN}
ROUND_NUM=${5:-$DEFAULT_ROUND_NUM}
CONFIG_NAME=${6:-$DEFAULT_CONFIG_NAME}
COUPLING_DIMS=${7:-$DEFAULT_COUPLING_DIMS}
NOISE_LEVEL=${8:-$DEFAULT_NOISE_LEVEL}
REGULARIZATION=${9:-$DEFAULT_REGULARIZATION}

python3 RUN.py pgclustered_esn \
--mode $MODE \
--display_output 0 \
--system_name Lorenz96_F8_ALPHA1_DIM40 \
--experiment_name Lorenz96_F8_ALPHA1_DIM40 \
--write_to_log 1 \
--N_train 60000 \
--N_test 40000 \
--RDIM 40 \
--scaler Standard \
--approx_reservoir_size $APPROX_RESERVOIR_SIZE \
--sparsity $SPARSITY \
--p_in $PIN \
--dynamics_length 2000 \
--iterative_prediction_length 7000 \
--num_test_ICS 5 \
--hyper_tuning_round_num $ROUND_NUM \
--hyper_tuning_config_name $CONFIG_NAME \
--loss d_temp \
--in_cluster_weight 0.5 \
--coupling_dims $COUPLING_DIMS \
--noise_level $NOISE_LEVEL \
--regularization $REGULARIZATION \
