#!/bin/bash

cd ../../Methods

python3 ANALYSE.py plot_mle_analysis \
--mode plot_mle_analysis \
--experiment_name Lorenz96_ANALYSIS \
--system_list Lorenz96_F8_ALPHA1_DIM40 Lorenz96_F8_ALPHA2_DIM40 Lorenz96_F8_ALPHA3_DIM40 Lorenz96_F8_ALPHA4_DIM40 Lorenz96_F8_ALPHA5_DIM40 Lorenz96_F8_ALPHA6_DIM40 Lorenz96_F8_ALPHA7_DIM40 Lorenz96_F8_ALPHA8_DIM40 Lorenz96_F8_ALPHA9_DIM40 Lorenz96_F8_ALPHA10_DIM40 \
--x_value ALPHA \
--x_value_list  1 2 3 4 5 6 7 8 9 10\