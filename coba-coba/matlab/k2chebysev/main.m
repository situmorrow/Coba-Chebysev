% CHEBYSHEV NMPC: Pure MATLAB Implementation
% Point stabilization for ship using Chebyshev Pseudospectral Method
% Requirements: MATLAB Optimization Toolbox (fmincon)

clc; clear all; close all;

% Check for required toolbox
if ~exist('fmincon', 'file')
    error('MATLAB Optimization Toolbox is required but not installed.');
end

%% Setup
fprintf('=== Chebyshev NMPC Ship Control ===\n\n');

% Load ship parameters
params = setupShipParams();

% NMPC parameters
N = 20;                 % Prediction horizon (polynomial order)
T = 30;                 % Prediction horizon time (seconds)
time_sampling = 1;      % Sampling time (seconds)

% Generate Chebyshev nodes and matrices
fprintf('Generating Chebyshev nodes and matrices... ');
tau = chebyshevNodes(N);        % (N+1)x1 nodes in [-1,1]
D = chebyshevDiffMatrix(tau);   % (N+1)x(N+1) differentiation matrix
w = chebyshevWeights(N);        % (N+1)x1 quadrature weights
fprintf('Done.\n');

% Simulation parameters
x0 = [0; 0; 0; 0; pi/2];        % Initial state: [v; r; x; y; psi]
reference_pose = [500; 2000; 0]; % Reference: [x; y; psi]
simulation_time = 300;          % Max simulation time (s)
distance_threshold = 10;        % Stopping distance (m)

% Initial storage
state_history = x0;
control_history = 0;
time_history = 0;
solve_times = zeros(1, simulation_time);

%% Simulation Loop
mpc_iter = 0;
distance_to_goal = norm(x0(3:4) - reference_pose(1:2));
fprintf('Initial distance to goal: %.2f m\n', distance_to_goal);
fprintf('Starting MPC loop...\n\n');

while distance_to_goal > distance_threshold && mpc_iter < simulation_time/time_sampling
    % Solve Chebyshev NMPC
    tic;
    [u_opt, pred_states, info] = chebyshevNMPCSolver(...
        x0, reference_pose, params, N, T, tau, D, w);
    solve_time = toc;
    
    % Store solve time
    solve_times(mpc_iter+1) = solve_time;
    
    % Check solver status
    if info.exitflag <= 0
        fprintf('Warning: Solver failed at iteration %d (exitflag: %d)\n', ...
            mpc_iter, info.exitflag);
    end
    
    % Apply control and simulate
    u_current = u_opt;
    xdot = shipModel(x0, u_current, params);
    x_next = x0 + time_sampling * xdot;
    
    % Update state
    x0 = x_next;
    
    % Store results
    state_history = [state_history, x0];
    control_history = [control_history, u_current];
    time_history = [time_history, mpc_iter * time_sampling];
    
    % Update distance
    distance_to_goal = norm(x0(3:4) - reference_pose(1:2));
    
    % Display progress
    if mod(mpc_iter, 15) == 0
        fprintf('Iter %3d | Pos=(%6.1f,%6.1f) | Dist=%8.2f | Ctrl=%6.2f° | Time=%6.3fs\n', ...
            mpc_iter, x0(3), x0(4), distance_to_goal, rad2deg(u_current), solve_time);
    end
    
    mpc_iter = mpc_iter + 1;
end

fprintf('\n=== Simulation Complete ===\n');
fprintf('Final position: (%.2f, %.2f)\n', x0(3), x0(4));
fprintf('Final distance to goal: %.2f m\n', distance_to_goal);
fprintf('Average solve time: %.4f s\n', mean(solve_times(1:mpc_iter)));
fprintf('Total iterations: %d\n', mpc_iter);

%% Plotting
figure('Name', 'Ship Trajectory', 'Position', [100 100 1200 500]);

% Trajectory plot
subplot(1,2,1);
plot(state_history(3,:), state_history(4,:), '-b', 'LineWidth', 2);
hold on;
plot(reference_pose(1), reference_pose(2), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('X Position (m)', 'FontSize', 12);
ylabel('Y Position (m)', 'FontSize', 12);
title('Ship Path', 'FontSize', 14);
legend('Trajecttory', 'Reference', 'Location', 'best');
grid on;
axis equal;

% State plots
subplot(1,2,2);
yyaxis left;
plot(time_history, rad2deg(state_history(5,:)), '-b', 'LineWidth', 1.5);
ylabel('Heading (deg)', 'FontSize', 12);
yyaxis right;
stairs(time_history(2:end), rad2deg(control_history(2:end)), '-k', 'LineWidth', 1.5);
ylabel('Rudder Angle (deg)', 'FontSize', 12);
xlabel('Time (s)', 'FontSize', '12');
title('Heading and Control', 'FontSize', 14);
grid on;

% Detailed control plot
figure('Name', 'Control and Rates', 'Position', [100 600 1200 400]);
subplot(2,1,1);
stairs(time_history(2:end), rad2deg(control_history(2:end)), '-k', 'LineWidth', 1.5);
hold on;
yline(rad2deg(params.u_max), '--r');
yline(rad2deg(params.u_min), '--r');
ylabel('Rudder Angle (°)', 'FontSize', 12);
title('Control Input and Constraints', 'FontSize', 14);
legend('Rudder', 'Constraints', 'Location', 'best');
grid on;

subplot(2,1,2);
r_rate = diff(control_history) / time_sampling;
plot(time_history(2:end-1), rad2deg(r_rate), '-m', 'LineWidth', 1.5);
hold on;
yline(rad2deg(params.rrot_max), '--r');
yline(rad2deg(params.rrot_min), '--r');
ylabel('Rudder Rate (°/s)', 'FontSize', 12);
xlabel('Time (s)', 'FontSize', 12);
legend('Rate', 'Constraints', 'Location', 'best');
grid on;

% Save results
results = [time_history; state_history; control_history];
writematrix(results', 'chebyshev_nmpc_results.csv');
fprintf('Results saved to: chebyshev_nmpc_results.csv\n');