#!/bin/bash

cd ../../../Methods

# Default values
DEFAULT_MODE="train"
DEFAULT_RESERVOIR_SIZE=1000
DEFAULT_SPARSITY=1
DEFAULT_PIN=1
DEFAULT_ROUND_NUM=0
DEFAULT_CONFIG_NAME="grid_search.json"
DEFAULT_PARALLEL_GROUP_SIZE=8
DEFAULT_PARALLEL_GROUP_INTERACTION_LENGTH=8
DEFAULT_N_TRAIN=100000
DEFAULT_NOISE_LEVEL=5
DEFAULT_REGULARIZATION=1e-2


# Read parameters
MODE=${1:-$DEFAULT_MODE}
APPROX_RESERVOIR_SIZE=${2:-$DEFAULT_RESERVOIR_SIZE}
ROUND_NUM=${3:-$DEFAULT_ROUND_NUM}
CONFIG_NAME=${4:-$DEFAULT_CONFIG_NAME}
PARALLEL_GROUP_SIZE=${5:-$DEFAULT_PARALLEL_GROUP_SIZE}
PARALLEL_GROUP_INTERACTION_LENGTH=${6:-$DEFAULT_PARALLEL_GROUP_INTERACTION_LENGTH}
N_TRAIN=${7:-$DEFAULT_N_TRAIN}
NOISE_LEVEL=${8:-$DEFAULT_NOISE_LEVEL}
REGULARIZATION=${9:-$DEFAULT_REGULARIZATION}

python3 RUN.py paralleled_esn \
--mode $MODE \
--display_output 0 \
--system_name KuramotoSivashinsky_INHOMO_MU01_L100_DIM256 \
--experiment_name KuramotoSivashinsky_INHOMO_MU01_L100_DIM256 \
--write_to_log 1 \
--N_test 100000 \
--RDIM 256 \
--scaler Standard \
--approx_reservoir_size $APPROX_RESERVOIR_SIZE \
--sparsity 1 \
--p_in 1 \
--dynamics_length 2000 \
--iterative_prediction_length 7000 \
--num_test_ICS 10 \
--worker_id 0 \
--hyper_tuning_round_num $ROUND_NUM \
--hyper_tuning_config_name $CONFIG_NAME \
--parallel_group_size $PARALLEL_GROUP_SIZE \
--parallel_group_interaction_length $PARALLEL_GROUP_INTERACTION_LENGTH \
--N_train $N_TRAIN \
--noise_level $NOISE_LEVEL \
--regularization $REGULARIZATION \