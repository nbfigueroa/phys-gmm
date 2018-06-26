%% Plot cosine similarity value

Xi = [0;1]
Xi_all = [0 1 0 -1; 
          1 1 -1 -1]

xi_dot_i = Xi;
for j=1:length(Xi_all)
    % Compute Velocity component
    % Normalize vectors    
    xi_dot_j = Xi_all(:,j)/norm(Xi_all(:,j));
    % Compute Angle
    s_angle = atan2(xi_dot_i(2),xi_dot_i(1))-atan2(xi_dot_j(2),xi_dot_j(1));
    
    % Compute shifted cosine of angle
    cos_angle = cos(s_angle);
    if isnan(cos_angle)
        cos_angle = 0;
    end
    s = 1 + cos_angle;
    
    % Compute Position component
    xi_i = Xi_ref(:,i);
    xi_j = Xi_ref(:,j);
    % LASA DATASET
    p = exp(-0.001*norm(xi_i - xi_j));
    p = 1;
    % GUI DATASET
    %                 p = exp(-1*norm(xi_i - xi_j));
    
    % Shifted Cosine Similarity of velocity vectors
    S(i,j) = p*s;
    
end
