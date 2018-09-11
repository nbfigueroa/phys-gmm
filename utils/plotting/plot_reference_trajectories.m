function [h_data, h_vel] = plot_reference_trajectories(Data, vel_sample)

% Plot the position trajectories
h_data = plot(Data(1,:),Data(2,:),'r.','markersize',10); hold on;

% Plot Velocities of Reference Trajectories
vel_points = Data(:,1:vel_sample:end);
U = zeros(size(vel_points,2),1);
V = zeros(size(vel_points,2),1);
for i = 1:size(vel_points, 2)
    dir_    = vel_points(3:end,i)/norm(vel_points(3:end,i));
    U(i,1)   = dir_(1);
    V(i,1)   = dir_(2);
end
h_vel = quiver(vel_points(1,:)',vel_points(2,:)', U, V, 0.5, 'Color', 'k', 'LineWidth',2); hold on;

end