#!/bin/bash

cd ../../Methods


python3 ANALYSE.py plot_mle_analysis \
--mode plot_mle_analysis \
--experiment_name Lorenz96_ANALYSIS \
--system_list Lorenz96_F8_ALPHA1_DIM40 Lorenz96_F12_ALPHA1_DIM40 Lorenz96_F16_ALPHA1_DIM40 Lorenz96_F20_ALPHA1_DIM40 Lorenz96_F24_ALPHA1_DIM40 Lorenz96_F28_ALPHA1_DIM40 Lorenz96_F32_ALPHA1_DIM40 Lorenz96_F36_ALPHA1_DIM40 Lorenz96_F40_ALPHA1_DIM40 Lorenz96_F44_ALPHA1_DIM40 \
--x_value F \
--x_value_list  8 12 16 20 24 28 32 36 40 44 \