%% Plot a Covariance Feature
figure('Color',[1 1 1])
imagesc(sigmas{1})

colormap(vivid); colorbar
axis square
title('Sample Youtube Covariance Feature','Interpreter','Latex','FontSize',16)
