%% Generate and Plot Data
close all; clear all; clc;
d = 2;
k = 3;
n = 500;
[X,label] = mixGaussRnd(d,k,n);
plotClass(X,label);

%% Collapse Gibbs sampling for Dirichelt process gaussian mixture model
tic;
[y,Theta,w,ll] = mixGaussGb(X);
toc;
figure
plotClass(X,y);