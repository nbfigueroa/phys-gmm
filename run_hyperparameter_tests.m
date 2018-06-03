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

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Compute Similarity Matrix from B-SPCM Function for dataset   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
steps = 20;
Sampler_Stats = [];
tau_range   = logspace(log10(0.1),log10(50),steps);
alpha_range = logspace(log10(0.1),log10(20),steps);

% %%%%%% Compute Confusion Matrix of Similarities %%%%%%%%%%%%%%%%%%
for i_tau=1:steps
    tau = tau_range(i_tau);
    spcm = ComputeSPCMfunctionMatrix(sigmas, tau);
    S = spcm(:,:,2);
    
    %%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
%     if exist('h0','var') && isvalid(h0), delete(h0); end
%     title_str = 'Bounded Similarity Function (B-SPCM) Matrix';
%     h0 = plotSimilarityConfMatrix(S, title_str);
    
    %%%%%%%%%%% Automatic Discovery of Dimensionality on M Manifold %%%%%%%%%%%
    M = [];
    [Y, d, thres, V] = spectral_DimRed(S, M);
    if isempty(M)
        s_norm = normalize_soft(softmax(d));
        M = sum(s_norm <= thres);
    end
    
    %%%%%%%% Visualize Spectral Manifold Representation for M=2 or M=3 %%%%%%%%
%     if exist('h1','var') && isvalid(h1), delete(h1);end
%     h1 = plotSpectralManifold(Y, true_labels, d,thres, s_norm, M);
    
    for iter=1:steps
        alpha = alpha_range(iter)
        %%%%%%%% Non-parametric Clustering on Manifold Data with Sim prior %%%%%%%%
        % Setting sampler/model options (i.e. hyper-parameters, alpha, Covariance matrix)
        options                 = [];
        options.type            = 'full';             % Type of Covariance Matrix: 'full' = NIW or 'Diag' = NIG
        options.T               = 200;                % Sampler Iterations
        options.alpha           = alpha;  % Concentration parameter
        
        % Standard Base Distribution Hyper-parameter setting
        if strcmp(options.type,'diag')
            lambda.alpha_0       = M;                    % G(sigma_k^-1|alpha_0,beta_0): (degrees of freedom)
            lambda.beta_0        = sum(diag(cov(Y')))/M; % G(sigma_k^-1|alpha_0,beta_0): (precision)
        end
        if strcmp(options.type,'full')
            lambda.nu_0        = M;                           % IW(Sigma_k|Lambda_0,nu_0): (degrees of freedom)
            lambda.Lambda_0    = eye(M)*sum(diag(cov(Y')))/M; % IW(Sigma_k|Lambda_0,nu_0): (Scale matrix)
%             lambda.Lambda_0    = diag(diag(cov(Y')));         % IW(Sigma_k|Lambda_0,nu_0): (Scale matrix)
        end
        lambda.mu_0             = mean(Y,2);    % hyper for N(mu_k|mu_0,kappa_0)
        %     lambda.mu_0             = zeros(size(Y(:,1)));    % hyper for N(mu_k|mu_0,kappa_0)
        lambda.kappa_0          = 1;            % hyper for N(mu_k|mu_0,kappa_0)
        
        
        % Run Collapsed Gibbs Sampler
        options.lambda    = lambda;
        [Psi Psi_Stats]   = run_ddCRP_sampler(Y, S, options);
        est_labels        = Psi.Z_C';
        
        
        Sampler_Stats(i_tau,iter).Psi = Psi;
        Sampler_Stats(i_tau,iter).Psi_Stats = Psi_Stats;
    end
end

%% %%%%%% Compute Collapsed Gibbs Sampler Clustering vs Ground Truth %%%%%%%%%%%%%%
steps = 20;
cluster_purity = zeros(steps,steps);
cluster_NMI    = zeros(steps,steps);
cluster_F      = zeros(steps,steps);
MaxLogProbs    = zeros(steps,steps);

for i_tau=1:steps
    for iter=1:steps
    Psi       = Sampler_Stats(i_tau,iter).Psi;
    est_labels = Psi.Z_C  ;  
    [cluster_purity(i_tau,iter) cluster_NMI(i_tau,iter) cluster_F(i_tau,iter)] = cluster_metrics(true_labels, est_labels');    
    MaxLogProbs(i_tau,iter) = Psi.MaxLogProb;
    end
end

%% %%%%%% Visualize Collapsed Gibbs Sampler Clustering vs Ground Truth %%%%%%%%%%%%%%
figure('Color',[1 1 1]);
colormap hot;
x = alpha_range;
y = tau_range;
z = cluster_F;
contourf(x,y,z)

set(gca,'xscale','log')
set(gca,'yscale','log')
set(gca, 'XTick', alpha_range)
set(gca,'XTickLabel', cellstr(num2str(alpha_range(:), '%4.2f')))
set(gca,'XTickLabelRotation',45)
set(gca, 'YTick', tau_range)
set(gca,'YTickLabel', cellstr(num2str(tau_range(:), '%4.2f')))

title({'$\mathcal{F}$-Measure'},'Interpreter','LaTex','FontSize',20)
xlabel('Concentration Parameter ($\alpha$)','Interpreter','LaTex','FontSize',20);
ylabel('Tolerance Parameter ($\tau$)','Interpreter','LaTex','FontSize',20);
colorbar
grid off
axis square


figure('Color',[1 1 1]);
colormap hot;
x = alpha_range;
y = tau_range;
z = MaxLogProbs;
contourf(x,y,z)

set(gca,'xscale','log')
set(gca,'yscale','log')
set(gca, 'XTick', alpha_range)
set(gca,'XTickLabel', cellstr(num2str(alpha_range(:), '%4.2f')))
set(gca,'XTickLabelRotation',45)
set(gca, 'YTick', tau_range)
set(gca,'YTickLabel', cellstr(num2str(tau_range(:), '%4.2f')))

title({'Max log $p(C|Y,S, \alpha, \lambda)$'},'Interpreter','LaTex','FontSize',20)
xlabel('Concentration Parameter ($\alpha$)','Interpreter','LaTex','FontSize',20);
ylabel('Tolerance Parameter ($\tau$)','Interpreter','LaTex','FontSize',20);
colorbar
grid off
axis square

