# phys-gmm
This package contains the inference implementation (Collapsed Gibbs Sampler) for the "Physically Consistent Bayesian Non-Parametric Mixture Model" (PC-GMM) proposed in [1]. This approach is used to **automatically** (no model selection!) fit GMM on **trajectory data** while ensuring that the points clustered in each Gaussian represent/follow some linear dynamics.; i.e. not only should they be close in "Euclidean"-position space but they should also follow the same direction. 

<p align="center">
  <img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/Lshape_pcgmm.png" width="220">
<img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/Ashape_pcgmm.png" width="220"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/Sshape_pcgmm.png" width="220"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/Ashape_pcgmm.png" width="220">
</>


### Instructions
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
est_options.samplerIter      = 100;  % Maximum Sampler Iterations
                                    % For type 0: 20-50 iter is sufficient
                                    % For type 2: >100 iter are needed
                                    
est_options.do_plots         = 1;   % Plot Estimation Statistics
est_options.sub_sample       = 1;   % Size of sub-sampling of trajectories

% Metric Hyper-parameters
est_options.estimate_l       = 1;   % Estimate the lengthscale, if set to 1
est_options.l_sensitivity    = 2;   % lengthscale sensitivity [1-10->>100]
                                    % Default value is set to '2' as in the
                                    % paper, for very messy, close to
                                    % self-interescting trajectories, we
                                    % recommend a higher value
est_options.length_scale     = [];  % if estimate_l=0 you can define your own
                                    % l, when setting l=0 only
                                    % directionality is taken into account

% Fit GMM to Trajectory Data
[Priors, Mu, Sigma] = fit_gmm(Xi_ref, Xi_dot_ref, est_options);

```
To test the function, you can either draw 2D data by running the demo script:
```
demo_drawData.m
```
or you can load pre-drawn 2D or real 3D datasets with the following script:
```
demo_loadData.m
```
### Example Datasets
The following datasets are provided in ```
./datasets``` folder. Following we show fits from pc-gmm **(top right)**, EM fit with Model Selection via BIC score **(bottom left)** and Bayesian Non-Parametric inference via CRP formulation **(bottom right)**.

-  **GMM fit on 2D Concentric Circles Dataset**

<p align="center">
  <img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/concentric-data.png" width="220">
<img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/concentric-pcgmm.png" width="220"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/concentric-emgmm.png" width="220"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/concentric-crpgmm.png" width="220">
</>

- **GMM fit on 2D Snake Dataset**
<p align="center">
  <img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/sine-data.png" width="220">
<img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/sine-pcgmm.png" width="220"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/sine-emgmm.png" width="220"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/sine-crpgmm.png" width="220">
</>  

-  **GMM fit on 2D Via-Point Dataset**  
<p align="center">
  <img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/viapoint-data.png" width="220">
<img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/viapoint-pcgmm.png" width="220"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/viapoint-emgmm.png" width="220"><img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/viapoint-crpgmm.png" width="220">
</>  

-  **GMM fit on 3D Via-Point Dataset**  
TODO...

-  **GMM fit on 3D Bumpy Snake Dataset**  
TODO...

### Estimation Statistics
By setting ```est_options.do_plots= 1;``` the function will plot the corresponding estimation statistics for each algorithm. 
- For the PC-GMM we show the values of the posterior distribution p(C|...) and the estimated clusters at each iteration:  
<p align="center">
  <img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/example-PCGMM-stats.png" width="540">
</>  

- For the EM-based Model Selection approach we show the BIC curve computed with increasing K=1,...,15. The 1st and 2nd order numerical derivative of this curve is also plotted and the 'optimal' K is selected as the inflection point:  
<p align="center">
  <img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/example-BIC.png" width="540">
</>  

- For the CRP-GMM we show the values of the posterior distribution p(Z|...) and the estimated clusters at each iteration:  
<p align="center">
  <img src="https://github.com/nbfigueroa/phys-gmm/blob/master/figs/example-CRP-stats.png" width="540">
</>  

### Usage
Such physically-consistent clustering is particularly useful for learning Dynamical Systems (DS) that are formulated as Linear Parameter Varying (LPV) systems, as introduced in [1,2]. To use this approach for DS learning, download the [lpv-opt](https://github.com/nbfigueroa/lpv-opt.git) package.   

**References**    
> [1] Figueroa, N. and Billard, A. (2018) "A Physically-Consistent Bayesian Non-Parametric Mixture Model for Dynamical System Learning". In Proceedings of the 2nd Conference on Robot Learning (CoRL). Accepted.  
> [2] Mirrazavi Salehian, S. S. (2018) "Compliant control of Uni/ Multi- robotic arms with dynamical systems". PhD Thesis.  

**Contact**: [Nadia Figueroa](http://lasa.epfl.ch/people/member.php?SCIPER=238387) (nadia.figueroafernandez AT epfl dot ch)
