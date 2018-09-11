# phys-gmm
This package contains the inference implementation (Collapsed Gibbs Sampler) for the "Physically Consistent Bayesian Non-Parametric Mixture Model" (PC-GMM) proposed in [1]. This approach is used to fit GMM to trajectory data while ensuring that the points clustered in each Gaussian represent/follow some linear dynamics.; i.e. not only should they be close in "Euclidean"-position space but they should also follow the same direction. 

This package offers the physically-consistent GMM fitting approach, as well as examples and code for fitting GMM with standard EM approach and the Bayesian non-parametric approach following the Chinese Restaurant Process construction through the ```[Mu, Priors, Sigma] = fit_gmm()``` function by filling its options as follows:
```
%%%%%%%%%%%%%%%%%% GMM Estimation Algorithm %%%%%%%%%%%%%%%%%%%%%%
% 0: Physically-Consistent Non-Parametric (Collapsed Gibbs Sampler)
% 1: GMM-EM Model Selection via BIC
% 2: CRP-GMM (Collapsed Gibbs Sampler)
est_options = [];
est_options.type             = 0;   % GMM Estimation Alorithm Type   

% If algo 1 selected:
est_options.maxK             = 15;  % Maximum Gaussians for Type 1
est_options.fixed_K          = [];  % Fix K and estimate with EM for Type 1

% If algo 0 or 2 selected:
est_options.samplerIter      = 200;  % Maximum Sampler Iterations
                                    % For type 0: 20 iter is sufficient
                                    % For type 2: >100 iter are needed
                                    
est_options.do_plots         = 1;   % Plot Estimation Statistics
est_options.locality_scaling = 1;   % Scaling for the similarity to improve locality, Default=1
est_options.sub_sample       = 1;   % Size of sub-sampling of trajectories

% Fit GMM to Trajectory Data
[Priors0, Mu0, Sigma0] = fit_gmm(Xi_ref, Xi_dot_ref, est_options);
```
To test the function, you can either draw 2D data by running the demo script:
```
demo_drawData.m
```
or you can load pre-drawn datasets with the following script:
```
demo_loadData.m
```
### Example Datasets
The following datasets are provided in ```
./datasets``` folder. Following we show fits from pc-gmm ***(top right)***, EM fit with Model Selection via BIC score ***(bottom left)*** and Bayesian Non-Parametric inference via CRP formulation ***(bottom right)**.
- ***GMM fit on Snake Dataset***
<p align="center">
  <img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/sine-data.png" width="300">
<img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/sine-pcgmm.png" width="300"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/sine-emgmm.png" width="300"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/sine-crpgmm.png" width="300">
</>

-  ***GMM fit on Concentric Circles Dataset***

<p align="center">
  <img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/concentric-data.png" width="300">
<img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/concentric-pcgmm.png" width="300"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/concentric-emgmm.png" width="300"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/concentric-crpgmm.png" width="300">
</>


Such physically-consistent clustering is particularly useful for learning Dynamical Systems (DS) that are formulated as Linear Parameter Varying (LPV) systems, as introduced in [1,2]. To use this approach to learn stable DS, you should download the [lpv-opt](https://github.com/nbfigueroa/lpv-opt.git)  package.   



**References**    
[1] Figueroa, N. and Billard, A. (2018) A Physically-Consistent Bayesian Non-Parametric Mixture Model for Dynamical System Learning. In Proceedings of the 2nd Conference on Robot Learning (CoRL). Accepted.  
[2] Mirrazavi Salehian, S. S. (2018) Compliant control of Uni/ Multi- robotic arms with dynamical systems. PhD Thesis.  

**Contact**: [Nadia Figueroa](http://lasa.epfl.ch/people/member.php?SCIPER=238387) (nadia.figueroafernandez AT epfl dot ch)
