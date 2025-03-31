#!/bin/bash

cd ../../Methods

DEFAULT_RESERVOIR_SIZE=1000
APPROX_RESERVOIR_SIZE=${1:-$DEFAULT_RESERVOIR_SIZE}

python3 ANALYSE.py plot_all_nonlinearity_and_icw_analysis_evaluation_results \
--mode plot_all_nonlinearity_and_icw_analysis_evaluation_results \
--model pgclustered_esn \
--experiment_name Lorenz96_ANALYSIS \
--icw_list 0.1 0.3 0.5 0.7 0.9 \
--parse_string  "round_(\d+)_grid-search-for-nonlinearity&icw_(\d+\.\d+)_reservoir_size_1000-pgclustered_esn_system_name_Lorenz96_F8_ALPHA(\d+)_DIM40_reservoir_size_1000_NTrain_100000_Noise_5_Regularization_(\d+\.\d+)" \
--icw_value_group_num 2 \
--x_value Alpha \
--x_value_group_num 3 \
--approx_reservoir_size $APPROX_RESERVOIR_SIZE \