# physics_guided_clustered_ESN


This is an implementation for our paper <a href="https://arxiv.org/abs/2504.01532" target="_blank">Incorporating Coupling Knowledge into Echo State Networks for Learning Spatiotemporally Chaotic Dynamics</a> [1].
This project contains implementations of echo state network (ESN) architectures for reconstruct high-dimensional dynamical systems. The following models are implemented:
- Standard ESN 
- paralleled ESN
- physics-guided clustered ESN (PGC-ESN)
- randomly clustered ESN (RandC-ESN)
- moved PGC-ESN (MovedPGC-ESN)
- partially PGC-ESN (PartPGC-ESN)

Standard ESN and spatial parallelization of the paralleled ESN are implemented according to [2].
PGC-ESN is our proposed model incorporating the coupling knowledge in a target system into its reservoir layer. If the coupling is asymmetric, we call the corresponding model as asymmetric PGC-ESN.
RandC-ESN is a comparison model proposed to check whether the clustered structure without incorporating prior coupling knowledge can improve dynamics reconstruction.
MovedPGC-ESN is a model after moving the coupling in the reservoir layer of PGC-ESN to outer wrong place.
PartPGC-ESN is a model after randomly rewiring the couplings in the PGC-ESN. These two models are used to check whehter imperfect coupling knowledge can enhance ESN's dynamics reconstruction performance.



## Code Requirements

- python 3.7.3 (or newer version)
- pytorch 1.5.1 (or newer version)
- matplotlib, sklearn, psutil
- mpi4py (parallel implementations)


## Parallel architectures

Parallelized networks that take advantage of the local interactions in the state space employ MPI communication.
After installing an MPI library (open-mpi or mpich), the mpi4py library can be installed with:
```
pip3 install mpi4py
```


## Datasets

The data to run a small demo are provided in the local ./Data folder


## Demo

In order to run the demo in a local cluster, you can navigate to the Experiments folder, and select your desired application, e.g. Lorenz3D_SIGMA10_RHO28_BETA2.67. There are scripts for each model. Please use the jupeter files
to run the .sh files.


## Contact info
For questions or to get in contact please refer to kueijanchu@outlook.com

[1] Chu, K. J., Akashi, N., & Yamamoto, A. (2025). Incorporating Coupling Knowledge into Echo State Networks for Learning Spatiotemporally Chaotic Dynamics. arXiv preprint arXiv:2504.01532.

[2] P.R. Vlachas, J. Pathak, B.R. Hunt et al., *Backpropagation algorithms and
Reservoir Computing in Recurrent Neural Networks for the forecasting of complex spatiotemporal
dynamics.*
Neural Networks, 2020 (doi: https://doi.org/10.1016/j.neunet.2020.02.016.)