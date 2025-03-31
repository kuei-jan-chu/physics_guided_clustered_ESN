#!/bin/bash

cd ../../../Methods

python3 RUN.py randomly_clustered_esn \
--mode test \
--display_output 1 \
--worker_id 0 \
--system_name Lorenz96_F8_ALPHA1_DIM40  \
--experiment_name Lorenz96_F8_ALPHA1_DIM40 \
--write_to_log 1 \
--N_train 100000 \
--N_test 100000 \
--RDIM 40 \
--noise_level 5 \
--scaler Standard \
--approx_reservoir_size 1000 \
--sparsity 0.1 \
--p_in 0.1 \
--radius 0.99 \
--sigma_input 0.5 \
--regularization 0.01 \
--dynamics_length 2000 \
--iterative_prediction_length 7000 \
--num_test_ICS 10 \

