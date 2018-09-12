%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo Code for GMM Learning for paper:                                   %
%  'A Physically-Consistent Bayesian Non-Parametric Mixture Model for     %
%   Dynamical System Learning.'                                           %
% With this script you can load 2D toy trajectories or even real-world 
% trajectories acquired via kinesthetic taching and test the different    %
% GMM fitting approaches.                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 1 (DATA LOADING): Load Datasets %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc
%%%%%%%%%%%% Select a Dataset %%%%%%%%%%%
% 1: Snake Dataset (2D)
% 2: Concentric Circle Dataset (2D)
% 3: Via-Point Dataset (2D)
% 4: Self-Intersecting Dataset (2D)
% 5: Scenario 1 - Viapoint (3D)
% 6: Scenario 2 - Sink (3D)
% 7: 
% 8: 
dataset = 1; 
Data = load_dataset(dataset);

%% Position/Velocity Trajectories
Xi_ref     = Data(1:2,:);
Xi_dot_ref = Data(3:end,:);
figure('Color',[1 1 1])
vel_samples = 15; vel_size = 0.85;
[h_data, h_vel] = plot_reference_trajectories(Data, vel_samples, vel_size);
grid on;
box on;
title('Reference Trajectories','Interpreter','LaTex','FontSize',20);
xlabel('$\xi_1$','Interpreter','LaTex','FontSize',20);
ylabel('$\xi_2$','Interpreter','LaTex','FontSize',20);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 2 (GMM FITTING): Fit GMM to Trajectory Data %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
est_options.samplerIter      = 20;  % Maximum Sampler Iterations
                                    % For type 0: 20 iter is sufficient
                                    % For type 2: >100 iter are needed
                                    
est_options.do_plots         = 1;   % Plot Estimation Statistics
est_options.locality_scaling = 1;   % Scaling for the similarity to improve locality, Default=1
est_options.sub_sample       = 2;   % Size of sub-sampling of trajectories

% Fit GMM to Trajectory Data
[Priors0, Mu0, Sigma0] = fit_gmm(Xi_ref, Xi_dot_ref, est_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 3 (FIT Visualization): Visualize Gaussian Components and labels %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract Cluster Labels
est_K      = length(Priors0); 
Priors = Priors0; Mu = Mu0; Sigma = Sigma0;
[~, est_labels] =  my_gmm_cluster(Xi_ref, Priors, Mu, Sigma, 'hard', []);

% Visualize Cluster Parameters on Manifold Data
plotGMMParameters( Xi_ref, est_labels, Mu, Sigma);
limits_ = limits + [-0.015 0.015 -0.015 0.015];
axis(limits_)
switch est_options.type   
    case 0
        title('Physically-Consistent Non-Parametric Mixture Model','Interpreter','LaTex', 'FontSize',15);
    case 1        
        title('Best fit GMM with EM-based BIC Model Selection','Interpreter','LaTex', 'FontSize',15);
    case 2
        title('Bayesian Non-Parametric Mixture Model (CRP-GMM)','Interpreter','LaTex', 'FontSize',15);
end
