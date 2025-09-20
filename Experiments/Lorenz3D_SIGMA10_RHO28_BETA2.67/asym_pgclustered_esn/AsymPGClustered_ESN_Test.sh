#!/bin/bash

cd ../../../Methods

python3 RUN.py asym_pgclustered_esn \
--mode test \
--display_output 1 \
--worker_id 0 \
--system_name Lorenz3D_SIGMA10_RHO28_BETA2.67  \
--experiment_name  Lorenz3D_SIGMA10_RHO28_BETA2.67  \
--write_to_log 1 \
--N_train 100000 \
--N_test 100000 \
--RDIM 3 \
--noise_level 5 \
--scaler Standard \
--approx_reservoir_size 200 \
--sparsity 1 \
--p_in 1 \
--radius 0.99 \
--sigma_input 1.0 \
--regularization 0.01 \
--dynamics_length 2000 \
--iterative_prediction_length 7000 \
--num_test_ICS 10 \
--in_cluster_weight 0.5 \
--coupling_dims "1 ; -1 1; -1 1" \

