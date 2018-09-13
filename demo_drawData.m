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
Data = [];
for l=1:length(data)    
    % Gather Data
    data_ = data{l};
    Data = [Data data_];
end

% Visualize Position/Velocity Trajectories
close;
Xi_ref     = Data(1:2,:);
Xi_dot_ref = Data(3:end,:);
vel_samples = 15; vel_size = 0.85;
[h_data, h_vel] = plot_reference_trajectories(Data, vel_samples, vel_size);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 2 (GMM FITTING): Fit GMM to Trajectory Data %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%% GMM Estimation Algorithm %%%%%%%%%%%%%%%%%%%%%%
% 0: Physically-Consistent Non-Parametric (Collapsed Gibbs Sampler)
% 1: GMM-EM Model Selection via BIC
% 2: CRP-GMM (Collapsed Gibbs Sampler)
est_options = [];
est_options.type             = 0;   % GMM Estimation Algorithm Type   

% If algo 1 selected:
est_options.maxK             = 15;  % Maximum Gaussians for Type 1
est_options.fixed_K          = [];  % Fix K and estimate with EM for Type 1

% If algo 0 or 2 selected:
est_options.samplerIter      = 50;  % Maximum Sampler Iterations
                                    % For type 0: 20 iter is sufficient
                                    % For type 2: >100 iter are needed
                                    
est_options.do_plots         = 1;   % Plot Estimation Statistics
est_options.locality_scaling = 0;   % Scaling for the similarity to improve locality, Default=1
est_options.sub_sample       = 2;   % Size of sub-sampling of trajectories

% Fit GMM to Trajectory Data
[Priors, Mu, Sigma] = fit_gmm(Xi_ref, Xi_dot_ref, est_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 3 (FIT Visualization): Visualize Gaussian Components and labels %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract Cluster Labels
est_K      = length(Priors); 
[~, est_labels] =  my_gmm_cluster(Xi_ref, Priors, Mu, Sigma, 'hard', []);

% Visualize Cluster Parameters Trajectory Data
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

% Visualize PDF of fitted GMM
 ml_plot_gmm_pdf(Xi_ref, Priors, Mu, Sigma, limits)  
switch est_options.type   
    case 0
        title('Physically-Consistent Non-Parametric Mixture Model','Interpreter','LaTex', 'FontSize',15);
    case 1        
        title('Best fit GMM with EM-based BIC Model Selection','Interpreter','LaTex', 'FontSize',15);
    case 2
        title('Bayesian Non-Parametric Mixture Model (CRP-GMM)','Interpreter','LaTex', 'FontSize',15);
end
