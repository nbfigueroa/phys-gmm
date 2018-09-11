%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo Code for GMM Learning for paper:                                   %
%  'A Physically-Consistent Bayesian Non-Parametric Mixture Model for     %
%   Dynamical System Learning.'                                           %
% With this script you can draw 2D toy trajectories and test different    %
% GMM fitting approaches.                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 1 (DATA LOADING): Draw 2D Trajectories with GUI %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc
fig1 = figure('Color',[1 1 1]);
limits = [-4 4 -4 4];
axis(limits)
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.25, 0.55, 0.2646 0.4358]);
grid on

% Draw Reference Trajectories
data = draw_mouse_data_on_DS(fig1, limits);
Data = []; x0_all = [];
for l=1:length(data)    
    % Check where demos end and shift
    data_ = data{l};
    Data = [Data data_];
    x0_all = [x0_all data_(1:2,1)];
end

% Position/Velocity Trajectories
close;
Xi_ref     = Data(1:2,:);
Xi_dot_ref = Data(3:end,:);
figure('Color',[1 1 1])
vel_samples = 5;
[h_data, h_vel] = plot_reference_trajectories(Data, vel_samples);
grid on;
box on;
title('Reference Trajectories','Interpreter','LaTex','FontSize',20);
xlabel('$\xi_1$','Interpreter','LaTex','FontSize',20);
ylabel('$\xi_2$','Interpreter','LaTex','FontSize',20);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 2 (GMM FITTING): Fit GMM to Trajectory Data %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% GMM Estimation Algorithm %%%%
% 0: Physically-Consistent Non-Parametric (Collapsed Gibbs Sampler)
% 1: GMM-EM Model Selection via BIC
% 2: CRP-GMM (Collapsed Gibbs Sampler)
est_options = [];
est_options.type             = 2;   % GMM Estimation Alorithm Type    
est_options.maxK             = 15;  % Maximum Gaussians for Type 1/2
est_options.do_plots         = 1;   % Plot Estimation Statistics
est_options.fixed_K          = [];  % Fix K and estimate with EM
est_options.locality_scaling = 1;   % Scaling for the similarity to improve locality
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
