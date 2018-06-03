function logPr = logPr_NormalInvWishart( Mu, invSigma, hyper )
% Mu, Sigma ~ NormInvWish( params, hyper )
% PP has fields
%  .precMu : precision of mean  (k in Murphy)
%  .Mu     :  mean of mean      (Mu in Murphy)
%  .ScaleMatrix :  DxD matrix   ( S  in Murphy)
%  .degFree :  degrees of freedom (v in Murphy)

% Mean Hyper-parameters
mu        = hyper.mu;     % mean
kappa     = hyper.kappa;  % precision

% Covariance Hyper-parameters
nu     = hyper.nu;     % degrees of freedom
Lambda = hyper.Lambda; % Scale Matrix 

logPrSigma = logPr_InvWishart( invSigma, nu, Lambda );

logPrMu = logPr_Gaussian( Mu, mu, kappa*invSigma );

logPr = logPrSigma + logPrMu;
end
