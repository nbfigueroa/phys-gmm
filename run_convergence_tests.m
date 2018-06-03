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


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Compute Similarity Matrix from B-SPCM Function for dataset   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 1; % [1, 100] Set higher for noisy data, Set 1 for ideal data 
% Datasets 1-3:  tau = 1;
% Datasets 4a/4b tau = 5;
% Datasets 4a/4b tau = 5;
% Dataset 6: tau = 1;

% %%%%%% Compute Confusion Matrix of Similarities %%%%%%%%%%%%%%%%%%
spcm = ComputeSPCMfunctionMatrix(sigmas, tau);  
S = spcm(:,:,2);

%%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h0','var') && isvalid(h0), delete(h0); end
title_str = 'Bounded Similarity Function (B-SPCM) Matrix';
h0 = plotSimilarityConfMatrix(S, title_str);

%%%%%%%%%%% Automatic Discovery of Dimensionality on M Manifold %%%%%%%%%%%
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
%%                        Run sampler T times                            %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 20;
Sampler_Stats = [];
for run=1:T    
    
    %%%%%%%% Non-parametric Clustering on Manifold Data with Sim prior %%%%%%%%
    % Setting sampler/model options (i.e. hyper-parameters, alpha, Covariance matrix)
    options                 = [];
    options.type            = 'full';  % Type of Covariance Matrix: 'full' = NIW or 'Diag' = NIG
    options.T               = 500;     % Sampler Iterations
    options.alpha           = 1;     % Concentration parameter
    
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
%     lambda.mu_0             = zeros(size(Y(:,1)));    % hyper for N(mu_k|mu_0,kappa_0)
    lambda.kappa_0          = 1;            % hyper for N(mu_k|mu_0,kappa_0)
    
    
    % Run Collapsed Gibbs Sampler
    options.lambda    = lambda;
    [Psi Psi_Stats]   = run_ddCRP_sampler(Y, S, options);
    est_labels        = Psi.Z_C';
    
    
    Sampler_Stats(run).Psi = Psi;
    Sampler_Stats(run).Psi_Stats = Psi_Stats;
end

%% %%%%%% Visualize Collapsed Gibbs Sampler Convergence %%%%%%%%%%%%%%
figure('Color',[1 1 1])
Iterations    = length(Sampler_Stats(1).Psi_Stats.LogLiks);
T = length(Sampler_Stats);
cluster_ests  = zeros(1,T);
runs = [1 2 3 4 5 9 10 11 12 13 14 15 16 17 18 19 20];
% runs = [1:20];
for i=1:length(runs)
    Psi_Stats = Sampler_Stats(runs(i)).Psi_Stats;
    Psi = Sampler_Stats(runs(i)).Psi;
    cluster_ests(i) = Psi_Stats.TotalClust(Psi.Maxiter); 
    semilogx(1:length(Psi_Stats.PostLogProbs),Psi_Stats.PostLogProbs,'--*', 'LineWidth',2,'Color',[rand rand rand]); hold on;
%     plot(1:length(Psi_Stats.PostLogProbs),Psi_Stats.PostLogProbs,'--*', 'LineWidth',2,'Color',[rand rand rand]); hold on;
end
xlim([1 Iterations])
xlabel('Gibbs Iteration','Interpreter','LaTex','Fontsize',20); ylabel('LogPr','Interpreter','LaTex','Fontsize',20)
title ({sprintf('Trace of Posterior Probabilities $p(C|Y,S, \\alpha, \\lambda)$ for %d runs, $\\overline{K}=$%2.1f (%2.1f)',[T mean(cluster_ests) std(cluster_ests)])}, 'Interpreter','LaTex','Fontsize',20)
grid on


%% %%%%%% Compute Collapsed Gibbs Sampler Clustering vs Ground Truth %%%%%%%%%%%%%%
cluster_purity = zeros(T,Iterations);
cluster_NMI    = zeros(T,Iterations);
cluster_F      = zeros(T,Iterations);
for i=1:T    
    Psi_Stats = Sampler_Stats(i).Psi_Stats;
    for j=1:Iterations
     est_labels = Psi_Stats.TableAssign(:,j);   
     [cluster_purity(i,j) cluster_NMI(i,j) cluster_F(i,j)] = cluster_metrics(true_labels, est_labels');    
    end
end

%% %%%%%% Visualize Collapsed Gibbs Sampler Clustering vs Ground Truth %%%%%%%%%%%%%%
figure('Color',[1 1 1])
for i=1:length(runs)
semilogx(1:length(cluster_purity),cluster_NMI(runs(i),:),'--*', 'LineWidth',2,'Color',[rand rand rand]); hold on;
end
xlabel('Gibbs Iteration','Interpreter','LaTex','Fontsize',20); ylabel('$\mathcal{F}$-Measure','Interpreter','LaTex','Fontsize',20)
title ({sprintf('Estimated Clusters vs. Ground Truth over %d runs',[T])}, 'Interpreter','LaTex','Fontsize',20)
xlim([1 Iterations])
grid on

%% %%%%%% Visualize Collapsed Gibbs Sampler Computation Time %%%%%%%%%%%%%%
figure('Color',[1 1 1])
overall_comp = zeros(1,T);
for i=1:T    
    Psi_Stats = Sampler_Stats(i).Psi_Stats;
    overall_comp(i) = sum(Psi_Stats.CompTimes);
    semilogx(1:length(Psi_Stats.CompTimes),Psi_Stats.CompTimes,'--*', 'LineWidth',2','Color',[rand rand rand]); hold on; 
end
xlim([1 Iterations])
xlabel('Gibbs Iteration','Interpreter','LaTex','Fontsize',20); ylabel('Computation Time (s)','Interpreter','LaTex','Fontsize',20);
title ({sprintf('Computation times per/iteration for %d runs',[T])}, 'Interpreter','LaTex','Fontsize',20)
grid on
