function [ handle ] = plotClusterParameters( Y, est_labels, Mu, Sigma, varargin );
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

if nargin == 5
    handle = varargin{1};    
else
    handle = figure('Color', [1 1 1]);
end
M = size(Y,1);

% Plot Gaussians on Projected Data
if (M == 2) || (M == 3)
    % Plot M-Dimensional Points of Spectral Manifold
    idx_label   = est_labels;
    pred_clust = length(unique(est_labels));
    
    if M==2    
        for jj=1:pred_clust
            clust_color = [rand rand rand];                                             
            scatter(Y(1,idx_label==jj),Y(2,idx_label==jj), 50, clust_color, 'filled'); hold on;
            plotGMM(Mu(:,jj), Sigma(:,:,jj), clust_color, 1);
            alpha(.5)
        end 
        xlabel('y_1');ylabel('y_2');
        colormap(hot)
        grid on
        title('\Sigma_i-s Respresented in 2-d Spectral space')
    end

    if M==3
        subplot(3,1,1)
        clust_color = zeros(length(pred_clust),3);
        for jj=1:pred_clust
            clust_color(jj,:) = [rand rand rand];
            scatter(Y(1,idx_label==jj),Y(2,idx_label==jj), 50, clust_color(jj,:), 'filled');hold on  
            plotGMM(Mu(1:2,jj), Sigma(1:2,1:2,jj), clust_color(jj,:), 1);
            alpha(.5)
        end
        xlabel('y_1');ylabel('y_2');
        axis auto
        colormap(hot)
        grid on
        title('\Sigma_i-s Respresented in 2-d [y_1-y_2] Spectral space', 'Fontsize',14)
        
        subplot(3,1,2)
        for jj=1:pred_clust
            scatter(Y(1,idx_label==jj),Y(3,idx_label==jj), 50, clust_color(jj,:), 'filled');hold on  
            plotGMM(Mu([1 3],jj), Sigma([1 3],[1 3],jj), clust_color(jj,:), 1);
            alpha(.5)
        end
        xlabel('y_1');ylabel('y_3');
        axis auto
        colormap(hot)
        grid on
        title('\Sigma_i-s Respresented in 2-d [y_1-y_3] Spectral space', 'Fontsize',14)
        
        subplot(3,1,3)
        for jj=1:pred_clust
            scatter(Y(2,idx_label==jj),Y(3,idx_label==jj), 50, clust_color(jj,:), 'filled');hold on  
            plotGMM(Mu(2:3,jj), Sigma(2:3,2:3,jj), clust_color(jj,:), 1);
            alpha(.5)
        end
        xlabel('y_2');ylabel('y_3');
        axis auto
        colormap(hot)
        grid on
        title('\Sigma_i-s Respresented in 2-d [y_2-y_3] Spectral space', 'Fontsize',14)
        
    end
end


end

