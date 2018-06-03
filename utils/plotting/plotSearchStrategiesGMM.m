function [hf] = plotSearchStrategiesGMM(GMM, labels)    
    hf = figure('Color',[1 1 1]); hold on;grid on;
    
    % GMM 
    Priors = GMM.Priors;
    Mu     = GMM.Mu;
    Sigma  = GMM.Sigma;
    
    [NewMu NewSigma]    = getGaussianSlice(Mu,Sigma,[4 5 6]);
    handles             = plot3dGaussian(Priors, NewMu,NewSigma );
    alpha               = rescale(Priors,min(Priors),max(Priors),0.1,0.8);    
    
    % Clustered Sigmas GMM
    colors = hsv(length(unique(labels)));
    
    for i=1:size(handles,1)
        set(handles(i),'FaceLighting','phong','FaceColor',colors(labels(i),:),'FaceAlpha',alpha(i),'AmbientStrength',0.1,'EdgeColor','none');
    end
    
    plotcube([0.5 0.7 0.05],[ -0.25 -0.35  -0.025],0.35,[1 1 1]);
    plotcube([0.03  0.06 0.025],[ 0.15 0.125 0.025],0.35,[0.0 1.0 0.0]);
    
    set(gca,'ZTick',[-0.02 0.08]);
    camlight
    
    axis equal;
    hold off;
    
    title('Gaussian Mixture Model (strategies)');
    set(gca,'FontSize',16);
end