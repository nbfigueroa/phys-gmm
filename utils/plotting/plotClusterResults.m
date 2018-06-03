function [ Purity NMI F handle ] = plotClusterResults( true_labels, est_labels, options)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

handle = figure('Color', [1 1 1]);
subplot(2,1,1)
imagesc(true_labels)
axis equal tight
colormap(pink)
title('True Labels', 'Interpreter','LaTex')

subplot(2,1,2)
imagesc(est_labels)
axis equal tight
colormap(pink)
est_clust = length(unique(est_labels));

[Purity NMI F] = cluster_metrics(true_labels, est_labels');

% Parse options
clust_type = options.clust_type;
if strcmp(clust_type, 'spcm-CRP-MM')
    Psi = options.Psi;             
    title_string = sprintf('Clustering from %s K=%d, Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f', clust_type, est_clust, Purity, NMI, F);
    fprintf('MAP Cluster estimate recovered at iter %d: %d\n', Psi.Maxiter, est_clust);
    fprintf('%s LP: %d and Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', clust_type, Psi.MaxLogProb, Purity, NMI, F);
end

title(title_string, 'Interpreter','LaTex')


end

