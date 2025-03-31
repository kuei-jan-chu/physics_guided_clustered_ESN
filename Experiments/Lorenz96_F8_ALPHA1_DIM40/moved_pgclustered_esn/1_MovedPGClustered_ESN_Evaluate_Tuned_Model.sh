#!/bin/bash

cd ../../../Methods


# Default values
DEFAULT_MODE="train"
DEFAULT_RESERVOIR_SIZE=1000
DEFAULT_SPARSITY=1
DEFAULT_PIN=1
DEFAULT_ROUND_NUM=0
DEFAULT_CONFIG_NAME="grid_search.json"
DEFAULT_N_TRAIN=100000
DEFAULT_NOISE_LEVEL=5
DEFAULT_REGULARIZATION=1e-2
DEFAULT_COUPLING_DIMS="-2 -1 1"
DEFAULT_MOVE_DIM=1

# Read parameters
MODE=${1:-$DEFAULT_MODE}
APPROX_RESERVOIR_SIZE=${2:-$DEFAULT_RESERVOIR_SIZE}
SPARSITY=${3:-$DEFAULT_SPARSITY}
PIN=${4:-$DEFAULT_PIN}
ROUND_NUM=${5:-$DEFAULT_ROUND_NUM}
CONFIG_NAME=${6:-$DEFAULT_CONFIG_NAME}
MOVE_DIM=${7:-$DEFAULT_MOVE_DIM}
COUPLING_DIMS=${8:-$DEFAULT_COUPLING_DIMS}
N_TRAIN=${9:-$DEFAULT_N_TRAIN}
NOISE_LEVEL=${10:-$DEFAULT_NOISE_LEVEL}
REGULARIZATION=${11:-$DEFAULT_REGULARIZATION}

python3 RUN.py moved_pgclustered_esn \
--mode $MODE \
--display_output 0 \
--worker_id 0 \
--system_name Lorenz96_F8_ALPHA1_DIM40 \
--experiment_name Lorenz96_F8_ALPHA1_DIM40 \
--write_to_log 1 \
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
--initialization_num 10 \
--in_cluster_weight 0.5 \
--move_dim $MOVE_DIM \
--coupling_dims $COUPLING_DIMS \
--N_train $N_TRAIN \
--noise_level $NOISE_LEVEL \
--regularization $REGULARIZATION \