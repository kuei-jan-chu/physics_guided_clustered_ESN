#!/bin/bash

cd ../../../Methods

mpiexec --oversubscribe -n 40 python3 RUN.py paralleled_esn \
--mode test \
--display_output 1 \
--worker_id 0 \
--experiment_name Lorenz96_F8_ALPHA1_DIM40 \
--system_name Lorenz96_F8_ALPHA1_DIM40 \
--write_to_log 1 \
--N_train 100000 \
--N_test 100000 \
--RDIM 40 \
--noise_level 5 \
--scaler Standard \
--approx_reservoir_size 2000 \
--sparsity 1 \
--p_in 1 \
--radius 0.9 \
--sigma_input 0.1 \
--regularization 0.01 \
--dynamics_length 2000 \
--iterative_prediction_length 7000 \
--num_test_ICS 10 \
--parallel_group_size 1 \
--parallel_group_interaction_length 1 \
