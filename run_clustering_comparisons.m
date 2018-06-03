%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main demo script for the SPCM-CRP-MM Clustering Algorithm proposed in:
%
% N. Figueroa and A. Billard, “Transform-Invariant Clustering of SPD Matrices 
% and its Application on Joint Segmentation and Action Discovery}”
% Arxiv, 2017. 
%
% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch  
% Website: http://lasa.epfl.ch
% November 2016; Last revision: 18-February-2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%% Data Loading Parameter Description %%%%%%%%%%%%%%%%%%%%%%
% display:   [0,1]  -- Display Covariance matrices in their own format
% randomize: [0,1]  -- Randomize the Covariance Matrices indices
% split:     [1,10] -- Selected Data Split from ETH80 or Youtube Dataset
% type:      {'real', 'synthetic'} -- Type for DT-MRI Dataset
% data_path:  {'./data/'} -- Path to data folder
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                    --Select a Dataset to Test--                       %%    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1) Toy 3D dataset, 5 Samples, 2 clusters (c1:3, c2:2)
% This function loads the 3-D ellipsoid dataset used to generate Fig. 3, 4 
% and 5 from Section 4 and the results in Section 7 in the accompanying paper.

clc; clear all; close all;
display = 1; randomize = 0; dataset_name = 'Toy 3D';
[sigmas, true_labels] = load_toy_dataset('3d', display, randomize);

%% 2)  Toy 6D dataset, 60 Samples, 3 clusters (c1:20, c2:20, c3: 20)
% This function loads the 6-D ellipsoid dataset used to generate Fig. 6 and 
% from Section 4 and the results in Section 8 in the accompanying paper.

clc; clear all; close all;
display = 0; randomize = 0; dataset_name = 'Toy 6D';
[sigmas, true_labels] = load_toy_dataset('6d', display, randomize);

%% 3) Real 6D dataset, task-ellipsoids, 105 Samples, 3 clusters 
%% Cluster Distibution: (c1:63, c2:21, c3: 21)
% This function loads the 6-D task-ellipsoid dataset used to evaluate this 
% algorithm in Section 8 of the accompanying paper.
%
% Please cite the following paper if you make use of this data:
% El-Khoury, S., de Souza, R. L. and Billard, A. (2014) On Computing 
% Task-Oriented Grasps. Robotics and Autonomous Systems. 2015 

clc; clear all; close all;
data_path = './data/'; randomize = 0; dataset_name = 'Real 6D (Task-Ellipsoids)';
[sigmas, true_labels] = load_task_dataset(data_path, randomize);

%% 4a) Toy 3D dataset, Diffusion Tensors from Synthetic Dataset, 1024 Samples
%% Cluster Distibution: 4 clusters (each cluster has 10 samples)
% This function will generate a synthetic DW-MRI (Diffusion Weighted)-MRI
% This is done following the "Tutorial on Diffusion Tensor MRI using
% Matlab" by Angelos Barmpoutis, Ph.D. which can be found in the following
% link: http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
%
% To run this function you should download fanDTasia toolbox in the 
% ~/SPCM-CRP/3rdParty directory, this toolbox is also provided in 
% the tutorial link.

% clc; clear all; close all;
data_path = './data/'; type = 'synthetic'; display = 1; randomize = 0; 
[sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );
dataset_name = 'Synthetic DT-MRI';

%% 4b) Real 3D dataset, Diffusion Tensors from fanTDasia Dataset, 1024 Samples
%% Cluster Distibution: 4 clusters (each cluster has 10 samples)
% This function loads a 3-D Diffusion Tensor Image from a Diffusion
% Weight MRI Volume of a Rat's Hippocampus, the extracted 3D DTI is used
% to evaluate this algorithm in Section 8 of the accompanying paper.
%untitled
% To load and visualize this dataset, you must download the dataset files 
% in the  ~/SPCM-CRP/data directory. These are provided in the online 
% tutorial on Diffusion Tensor MRI in Matlab:
% http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
%
% One must also download the fanDTasia toolbox in the ~/SPCM-CRP/3rdParty
% directory, this toolbox is also provided in this link.

% clc; clear all; close all;
data_path = './data/'; type = 'real'; display = 1; randomize = 0; 
[sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );
dataset_name = 'Real DT-MRI';


%% 5) Real 400D dataset, Covariance Features from ETH80 Dataset, 40 Samples
%% Cluster Distibution: 8 classes/clusters (each cluster has 10 samples)
% This function loads the 400-D ETH80 Covariance Feature dataset 
% used to evaluate this algorithm in Section 8 of the accompanying paper.
%
%
% You must download this dataset from the following link: 
% http://ravitejav.weebly.com/classification-of-manifold-features.html
% and export it in the ~/SPCM-CRP/data directory
%
% Please cite the following paper if you make use of these features:
% R. Vemulapalli, J. Pillai, and R. Chellappa, “Kernel Learning for Extrinsic 
% Classification of Manifold Features”, CVPR, 2013. 

clc; clear all; close all;
data_path = './data/'; split = 1; randomize = 0; 
[sigmas, true_labels] = load_eth80_dataset(data_path, split, randomize);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Compute Similarity Matrix (S) and Spectral Embedding (Y)      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 1; % [1, 100] Set higher for noisy data, Set 1 for ideal data 
% Datasets 1-3:  tau = 1;
% Datasets 4a/4b tau = 5;
% Datasets 4a/4b tau = 5;

% %%%%%% Compute Confusion Matrix of Similarities %%%%%%%%%%%%%%%%%%
spcm = ComputeSPCMfunctionMatrix(sigmas, tau);  
S = spcm(:,:,2);

%%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h0','var') && isvalid(h0), delete(h0); end
title_str = 'Bounded Similarity Function (B-SPCM) Matrix';
h0 = plotSimilarityConfMatrix(S, title_str);

%% %%%%%%%%% Automatic Discovery of Dimensionality on M Manifold %%%%%%%%%%
M = [];
[Y, d, thres, V] = spectral_DimRed(S, M);
if isempty(M)
    s_norm = normalize_soft(softmax(d));
    M = sum(s_norm <= thres);
end

%%%%%%%% Visualize Spectral Manifold Representation for M=2 or M=3 %%%%%%%%
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSpectralManifold(Y, true_labels, d,thres, s_norm, M);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Run E-M Model Selection for GMM with 10 runs in a range of K     %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model Selection for GMM
K_range = [1:10]; repeats = 20; cov_type = 'full';
ml_gmm_eval(Y, K_range, repeats, cov_type)

%%  Compute GMM Stats with 'optimal' K
% Set "Optimal " GMM Hyper-parameters
K = 3; T = 10;
cluster_purity = zeros(1,T);
cluster_NMI    = zeros(1,T);
cluster_F      = zeros(1,T);
for run=1:T    
  [Priors, Mu, Sigma] = ml_gmmEM(Y, K);
  [est_labels] =  ml_gmm_cluster(Y, Priors, Mu, Sigma);
  % Compute Metrics
  [cluster_purity(run) cluster_NMI(run) cluster_F(run)] = cluster_metrics(true_labels, est_labels);
end

% Final Stats for Mixture Model
fprintf('*** Gaussian Mixture Model w/MS Results*** \n Clusters: %d Purity: %3.3f +- %3.3f \n NMI: %3.3f +- %3.3f --- F: %3.3f +- %3.3f \n',[K ...
    mean(cluster_purity) std(cluster_purity) mean(cluster_NMI) std(cluster_NMI) mean(cluster_F) std(cluster_F)])

%%%%%%%% Visualize Spectral Manifold Representation for M=2 or M=3 %%%%%%%%
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSpectralManifold(Y, est_labels, d,thres, s_norm, M);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Run Collapsed Gibbs Sampler for CRP-MM 10 times (Mo Chen's Implementation) %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 10;
cluster_purity = zeros(1,T);
cluster_NMI    = zeros(1,T);
cluster_F      = zeros(1,T);
est_clusters   = zeros(1,T);
for run=1:T  
    % Fit CRP Mixture Model to Data
    tic;
    [est_labels, Theta, w, ll] = mixGaussGb(Y);
    est_clusters(run)  = length(unique(est_labels));
    toc;
    % Compute Metrics
    [cluster_purity(run) cluster_NMI(run) cluster_F(run)] = cluster_metrics(true_labels, est_labels);    
end

%% Final Stats for CRP Mixture Model
fprintf('*** CRP Mixture Model (Mo Chen) Results*** \n Clusters: %3.3f +- %3.3f Purity: %3.3f +- %3.3f \n NMI: %3.3f +- %3.3f --- F: %3.3f +- %3.3f \n',[mean(est_clusters) std(est_clusters) ...
    mean(cluster_purity) std(cluster_purity) mean(cluster_NMI) std(cluster_NMI) mean(cluster_F) std(cluster_F)])

%%%%%%%% Visualize Spectral Manifold Representation for M=2 or M=3 %%%%%%%%
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSpectralManifold(Y, est_labels, d,thres, s_norm, M);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         Run Sampler for DP-MM 10 times (Frank Wood's Implementation)      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 10;
cluster_purity = zeros(1,T);
cluster_NMI    = zeros(1,T);
cluster_F      = zeros(1,T);
est_clusters   = zeros(1,T);
for run=1:T  
    % Fit CRP Mixture Model to Data
    iterations = 500;
    [class_id, mean_record, covariance_record, K_record, lP_record, alpha_record] = sampler(Y, iterations);
    [val , Maxiter]   = max(lP_record);
    est_labels        = class_id(:,Maxiter);
    est_clusters(run) = length(unique(est_labels));
    
    % Compute Metrics
    [cluster_purity(run) cluster_NMI(run) cluster_F(run)] = cluster_metrics(true_labels, est_labels);    
end

% Final Stats for CRP Mixture Model
fprintf('*** CRP Mixture Model (Frank Wood) Results*** \n Clusters: %3.3f +- %3.3f Purity: %3.3f +- %3.3f \n NMI: %3.3f +- %3.3f --- F: %3.3f +- %3.3f \n',[mean(est_clusters) std(est_clusters) ...
    mean(cluster_purity) std(cluster_purity) mean(cluster_NMI) std(cluster_NMI) mean(cluster_F) std(cluster_F)])

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        Run Collapsed Gibbs Sampler for SPCM-CRP 10 times              %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 10;
Sampler_Stats = [];
cluster_purity = zeros(1,T);
cluster_NMI    = zeros(1,T);
cluster_F      = zeros(1,T);
est_clusters   = zeros(1,T);
for run=1:T    
    
    %%%%%%%% Non-parametric Clustering on Manifold Data with Sim prior %%%%%%%%
    % Setting sampler/model options (i.e. hyper-parameters, alpha, Covariance matrix)
    options                 = [];
    options.type            = 'full';  % Type of Covariance Matrix: 'full' = NIW or 'Diag' = NIG
    options.T               = 200;     % Sampler Iterations
    options.alpha           = 1;       % Concentration parameter
    
    % Standard Base Distribution Hyper-parameter setting
    if strcmp(options.type,'diag')
        lambda.alpha_0       = M;                    % G(sigma_k^-1|alpha_0,beta_0): (degrees of freedom)
        lambda.beta_0        = sum(diag(cov(Y')))/M; % G(sigma_k^-1|alpha_0,beta_0): (precision)
    end
    if strcmp(options.type,'full')
        lambda.nu_0        = M;                           % IW(Sigma_k|Lambda_0,nu_0): (degrees of freedom)
        lambda.Lambda_0    = eye(M)*sum(diag(cov(Y')))/M; % IW(Sigma_k|Lambda_0,nu_0): (Scale matrix)
    end
    lambda.mu_0             = mean(Y,2);    % hyper for N(mu_k|mu_0,kappa_0)
    lambda.kappa_0          = 1;            % hyper for N(mu_k|mu_0,kappa_0)
    
    
    % Run Collapsed Gibbs Sampler
    options.lambda    = lambda;
    [Psi Psi_Stats]   = run_ddCRP_sampler(Y, S, options);
    est_labels        = Psi.Z_C';
    
    % Store Stats
    Sampler_Stats(run).Psi = Psi;
    Sampler_Stats(run).Psi_Stats = Psi_Stats;
    est_clusters(run)  = length(unique(est_labels));
    
    % Compute Metrics
    [cluster_purity(run) cluster_NMI(run) cluster_F(run)] = cluster_metrics(true_labels, est_labels');
end

%% Final Stats for SPCM-CRP Mixture Model
fprintf('*** SPCM-CRM Mixture Model Results*** \n Clusters: %3.3f +- %3.3f Purity: %3.3f +- %3.3f \n NMI: %3.3f +- %3.3f --- F: %3.3f +- %3.3f \n',[mean(est_clusters) std(est_clusters) ...
    mean(cluster_purity) std(cluster_purity) mean(cluster_NMI) std(cluster_NMI) mean(cluster_F) std(cluster_F)])

%%%%%%%% Visualize Spectral Manifold Representation for M=2 or M=3 %%%%%%%%
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSpectralManifold(Y, est_labels, d,thres, s_norm, M);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%% For Datasets 4a/b: Visualize cluster labels for DTI %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Visualize Estimated Cluster Labels as DTI
if exist('h3','var') && isvalid(h3), delete(h3);end
title = 'Estimated Cluster Labels of Diffusion Tensors';
h3 = plotlabelsDTI(est_labels, title);
