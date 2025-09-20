# physics_guided_clustered_ESN


This is an implementation for our paper <a href="https://doi.org/10.1063/5.0273343" target="_blank">Incorporating Coupling Knowledge into Echo State Networks for Learning Spatiotemporally Chaotic Dynamics</a> [1].
This project contains implementations of echo state network (ESN) architectures for reconstruct high-dimensional dynamical systems. The following models are implemented:
- Standard ESN 
- Paralleled ESN
- Physics-guided clustered ESN (PGC-ESN)
- Randomly clustered ESN (RandC-ESN)
- Partially PGC-ESN (PartPGC-ESN)
- Automatically PGC-ESN (AutoPGC-ESN)
- Asymmetrically PGC-ESN (AsymPGC-ESN)

Standard ESN and paralleled ESN are implemented according to the code repository of Ref. [2].
PGC-ESN is our proposed model incorporating the coupling knowledge in a target system into its reservoir layer. If the coupling is asymmetric, we call the model as asymmetrically PGC-ESN. If the coupling is extracted and used to guide the ESN model automaticlly, we call the model as automatically PGC-ESN.
If we randomly rewire the couplings in the PGC-ESN, we call the model as partially PGC-ESN. These two models are used to check whehter imperfect or extracted coupling knowledge can enhance ESN's dynamics reconstruction performance.
RandC-ESN is a comparison model proposed to check whether the clustered structure without incorporating coupling knowledge can improve dynamics reconstruction.



## Code Requirements

- python 3.7.3 
- pytorch 1.5.1 
- matplotlib, sklearn, psutil
- mpi4py (parallel implementations)



## Demo

In order to run the demo in your computer, you can select your desired application, e.g. Lorenz96_F8_ALPHA1_DIM40 in the Experiments folder. There are scripts for each model. Please use the jupeter notebooks to run the .sh files.


## Contact info
For questions or to get in contact please refer to kueijanchu@outlook.com

[1] Kuei-Jan Chu, Nozomi Akashi, Akihiro Yamamoto; Incorporating coupling knowledge into echo state networks for learning spatiotemporally chaotic dynamics.Chaos 1 September 2025; 35 (9): 093138. https://doi.org/10.1063/5.0273343

[2] P.R. Vlachas, J. Pathak, B.R. Hunt, T.P. Sapsis, M. Girvan, E. Ott, P. Koumoutsakos,
Backpropagation algorithms and Reservoir Computing in Recurrent Neural Networks for the forecasting of complex spatiotemporal dynamics,
Neural Networks,
Volume 126,
2020,
Pages 191-217,
ISSN 0893-6080,
https://doi.org/10.1016/j.neunet.2020.02.016.