function [X, z, model] = mixGaussRnd(d, k, n)
% Genarate samples form a Gaussian mixture model.
% Input:
%   d: dimension of data
%   k: number of components
%   n: number of data
% Output:
%   X: d x n data matrix
%   z: 1 x n response variable
%   model: model structure
% Written by Mo Chen (sth4nth@gmail.com).
alpha0 = 1;  % hyperparameter of Dirichlet prior
W0 = eye(d);  % hyperparameter of inverse Wishart prior of covariances
v0 = d+1;  % hyperparameter of inverse Wishart prior of covariances
mu0 = zeros(d,1);  % hyperparameter of Guassian prior of means
beta0 = nthroot(k,d); % hyperparameter of Guassian prior of means % in volume x^d there is k points: x^d=k


w = dirichletRnd(alpha0,ones(1,k)/k);
z = discreteRnd(w,n);

mu = zeros(d,k);
Sigma = zeros(d,d,k);
X = zeros(d,n);
for i = 1:k
    idx = z==i;
    Sigma(:,:,i) = iwishrnd(W0,v0); % invpd(wishrnd(W0,v0));
    mu(:,i) = gaussRnd(mu0,beta0*Sigma(:,:,i));
    X(:,idx) = gaussRnd(mu(:,i),Sigma(:,:,i),sum(idx));
end
model.mu = mu;
model.Sigma = Sigma;
model.w = w;



function x = gaussRnd(mu, Sigma, n)
% Generate samples from a Gaussian distribution.
% Input:
%   mu: d x 1 mean vector
%   Sigma: d x d covariance matrix
%   n: number of samples
% Outpet:
%   x: d x n generated sample x~Gauss(mu,Sigma)
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 2
    n = 1;
end
[V,err] = chol(Sigma);
if err ~= 0
    error('ERROR: sigma must be a symmetric positive definite matrix.');
end
x = V'*randn(size(V,1),n)+repmat(mu,1,n);