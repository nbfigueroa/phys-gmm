function [ sigmas, true_labels, GMM ] = load_search_dataset(data_path, type, display )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% Choose Dataset
switch type
    case 'full'
        load(strcat(data_path,'7D-Search-Strategies.mat'))
    case 'table'
        load(strcat(data_path,'7D-Search-Strategies-Surface.mat'))
end
    
% Create GMM Structure
GMM.Mu     = Mu;
GMM.Priors = Priors;
GMM.Sigma  = Sigma;

% Generate Sigma/labels
clear sigmas
for i=1:size(Sigma, 3)
    sigmas{i} = Sigma(:,:,i); 
end

clear true_labels
true_labels  = ones(1,length(sigmas));

% Plot GMM + Table + Axis
if display==1
    plotSearchStrategiesGMM(GMM, true_labels)
end


end

