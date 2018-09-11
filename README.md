# phys-gmm
This package contains the inference implementation (Gibbs Sampler) for the "Physically Consistent Bayesian Non-Parametric Mixture Model" (PC-GMM) proposed in [1]. This approach is used to fit GMM to trajectory data while ensuring that the points clustered in each Gaussian represent/follow some linear dynamics.; i.e. not only should they be close in "Eucliedan"-position space but they should also follow the same direction. 

<p align="center">
<img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/sine-pcgmm.png" width="300"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/sine-crpgmm.png" width="300"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/sine-crpgmm.png" width="300">
</>


This GMM fitting approach is useful for learning Dynamical Systems (DS) that are formulated as Linear Parameter Varying (LPV) systems, as introduced in [1,2]. This package offers only the physically-consistent GMM fitting approach, as well as examples and code for fitting GMM with standar EM approach and the Bayesian non-parametric approach following the Chinese Restaurant Process construction. To use this approach to learn stable DS, you should download the [lpv-opt](https://github.com/nbfigueroa/lpv-opt.git)  package.   




**References**    
[1] Figueroa, N. and Billard, A. (2018) A Physically-Consistent Bayesian Non-Parametric Mixture Model for Dynamical System Learning. In Proceedings of the 2nd Conference on Robot Learning (CoRL). Accepted.  
[2] Mirrazavi Salehian, S. S. (2018) Compliant control of Uni/ Multi- robotic arms with dynamical systems. PhD Thesis.  

**Contact**: [Nadia Figueroa](http://lasa.epfl.ch/people/member.php?SCIPER=238387) (nadia.figueroafernandez AT epfl dot ch)
