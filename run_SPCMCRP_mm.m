function [Psi Psi_Stats est_labels]  = run_SPCMCRP_mm(sigmas, clust_options)
% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = clust_options.tau; 
% %%%%%% Compute Confusion Matrix of Similarities %%%%%%%%%%%%%%%%%%
spcm = ComputeSPCMfunctionMatrix(sigmas, tau);  
S = spcm(:,:,2);
%%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if clust_options.plot_sim
    if exist('h0','var') && isvalid(h0), delete(h0); end
    title_str = 'Bounded Similarity Function (B-SPCM) Matrix';
    h0 = plotSimilarityConfMatrix(S, title_str);
end
%%%%%%%%%%% Automatic Discovery of Dimensionality on M Manifold %%%%%%%%%%%
M = [];
[Y, d, thres, V] = spectral_DimRed(S, M);
if isempty(M)
    s_norm = normalize_soft(softmax(d));
    M = sum(s_norm <= thres);
end

if size(Y,1) == 1
    M = 2;
    [Y, d, thres, V] = spectral_DimRed(S, M);
end
%%%%%%%% Non-parametric Clustering on Manifold Data with Sim prior %%%%%%%%

% Setting sampler/model options (i.e. hyper-parameters, alpha, Covariance matrix)
options                 = [];
options.type            = clust_options.type;  % Type of Covariance Matrix: 'full' = NIW or 'Diag' = NIG
options.T               = clust_options.T;     % Sampler Iterations 
options.alpha           = clust_options.alpha; % Concentration parameter

% Standard Base Distribution Hyper-parameter setting
if strcmp(options.type,'diag')
    lambda.alpha_0       = M;                    % G(sigma_k^-1|alpha_0,beta_0): (degrees of freedom)
    lambda.beta_0        = sum(diag(cov(Y')))/M; % G(sigma_k^-1|alpha_0,beta_0): (precision)
end
if strcmp(options.type,'full')
    lambda.nu_0        = M;                           % IW(Sigma_k|Lambda_0,nu_0): (degrees of freedom)
    lambda.Lambda_0    = eye(M)*sum(diag(cov(Y')))/M; % IW(Sigma_k|Lambda_0,nu_0): (Scale matrix)
%     lambda.Lambda_0    = diag(diag(cov(Y')));       % IW(Sigma_k|Lambda_0,nu_0): (Scale matrix)
end
lambda.mu_0             = mean(Y,2);    % hyper for N(mu_k|mu_0,kappa_0)
% lambda.mu_0             = zeros(size(Y(:,1)));    % hyper for N(mu_k|mu_0,kappa_0)
lambda.kappa_0          = 1;            % hyper for N(mu_k|mu_0,kappa_0)

% Run Collapsed Gibbs Sampler
options.lambda     = lambda;
options.init_clust = clust_options.init_clust;
if ~isfield(clust_options, 'verbose'); clust_options.verbose = 1;end
options.verbose    = clust_options.verbose;
[Psi Psi_Stats]    = run_ddCRP_sampler(Y, S, options);
est_labels         = Psi.Z_C';

end