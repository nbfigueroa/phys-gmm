function [] = plotlabelsaDTI(est_labels)

hf = figure('Color',[1 1 1]);
imagesc(flipud(reshape(est_labels,[sqrt(size(est_labels,2)) sqrt(size(est_labels,2))])))
colormap(pink)
colorbar
axis square

end