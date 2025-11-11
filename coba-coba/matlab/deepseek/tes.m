clc; clear; close all;

%% Parameters
% Simulation parameters
time_sampling = 1; % seconds
Np = 60; % prediction horizon (dikurangi untuk stabilitas)
simulation_time = 300;

% Ship constraints
r_max = 0.0932; r_min = -r_max;
u_max = deg2rad(35); u_min = -u_max;
rrot_max = deg2rad(5); rrot_min = -rrot_max;

% Cost weights
w_position = 1e-4;
w_orientation = 0.1;
w_control = 1e-2;

% Initial conditions
x0 = [0; 0; 0; 0; pi/2]; % [v; r; x; y; psi]
reference_pose = [500; 2000; 0]; % [x_ref; y_ref; psi_ref]

%% Ship Model Parameters (Corvette SIGMA class)
Lpp = 101.07; B = 14; T = 3.7; m = 2423 * 1e3;
os_surge = 15.4; CB = 0.65; xG = 5.25; rho = 1024;
Adelta = 5.7224; gyration = 0.156*Lpp;

% Hydrodynamics coefficients
Yvdot = -1*((1+0.16*CB*(B/T)-5.1*(B/Lpp)^2)*pi*(T/Lpp)^2);
Yrdot = -1*((0.67*(B/Lpp)-0.0033*(B/T)^2)*pi*(T/Lpp)^2);
Nvdot = -1*((1.1*(B/Lpp)-0.041*(B/T))*pi*(T/Lpp)^2);
Nrdot = -1*(((1/12)+0.017*(CB*B/T)-0.33*(B/Lpp))*pi*(T/Lpp)^2);
Yv = -1*((1+0.4*(CB*B/T))*pi*(T/Lpp)^2);
Yr = -1*((-0.5+2.2*(B/Lpp)-0.08*(B/T))*pi*(T/Lpp)^2);
Nv = -1*((0.5+2.4*(T/Lpp))*pi*(T/Lpp)^2);
Nr = -1*((0.25+0.039*(B/T)-0.56*(B/Lpp))*pi*(T/Lpp)^2);

Ydelta = rho*pi*Adelta/(4*Lpp*T);
Ndelta = -0.5*Ydelta;
Ir = (m*gyration^2)/(0.5*rho*Lpp^5);
Iz = (m*(xG^2))/(0.5*rho*Lpp^5)+Ir;

% Non-dimensional parameters
nd_u = 1; nd_m = m/(0.5*rho*Lpp^3); nd_xG = xG/Lpp;

% State space matrices
M = [nd_m-Yvdot, nd_m*nd_xG-Yrdot;
     nd_m*nd_xG-Nvdot, Iz-Nrdot];
N_mat = [-Yv, nd_m*nd_u-Yr;
         -Nv, nd_m*nd_xG*nd_u-Nr];
model_A = -M\N_mat;

a11 = model_A(1,1); a12 = model_A(1,2);
a21 = model_A(2,1); a22 = model_A(2,2);
model_B = [0.01; 1/Lpp];

% Operating point matrices
os_U = os_surge;
os_A = os_U*[a11, a12*Lpp; a21/Lpp, a22]/Lpp;
os_B = os_U^2*model_B/Lpp;

%% Chebyshev Pseudospectral Setup
N = 20; % Number of collocation points
[taus, D, weights] = chebyshev_setup(N);

% Time transformation parameters
t0 = 0; tf = Np * time_sampling;
T_scale = (tf - t0)/2;

%% Ship Dynamics Function
function xdot = ship_dynamics(x, u, os_A, os_B, os_surge)
    v = x(1); r = x(2); psi = x(5);
    
    % Dynamics for v and r
    x_dynamics = [v; r];
    xdot_dynamics = os_A * x_dynamics + os_B * u;
    
    % Full state derivatives
    xdot = [xdot_dynamics(1);                    % v_dot
            xdot_dynamics(2);                    % r_dot
            os_surge*cos(psi) - v*sin(psi);     % x_dot
            os_surge*sin(psi) + v*cos(psi);     % y_dot
            r];                                  % psi_dot
end

%% Chebyshev Setup Function - CORRECTED VERSION
function [taus, D, w] = chebyshev_setup(N)
    % Chebyshev-Gauss-Lobatto nodes
    taus = cos(pi*(0:N)/N);
    
    % Chebyshev differentiation matrix (corrected implementation)
    D = zeros(N+1, N+1);
    
    for i = 1:N+1
        for j = 1:N+1
            if i ~= j
                if i == 1 && j == 1
                    D(i,j) = (2*N^2 + 1)/6;
                elseif i == N+1 && j == N+1
                    D(i,j) = -(2*N^2 + 1)/6;
                else
                    if i == 1 || i == N+1
                        ci = 2;
                    else
                        ci = 1;
                    end
                    if j == 1 || j == N+1
                        cj = 2;
                    else
                        cj = 1;
                    end
                    D(i,j) = (ci/cj) * ((-1)^(i+j)) / (taus(i) - taus(j));
                end
            end
        end
    end
    
    % Diagonal elements
    for i = 2:N
        D(i,i) = -taus(i) / (2*(1 - taus(i)^2));
    end
    
    % Special cases for endpoints
    D(1,1) = (2*N^2 + 1)/6;
    D(N+1,N+1) = -(2*N^2 + 1)/6;
    
    % Chebyshev quadrature weights (simplified version)
    w = zeros(1, N+1);
    for i = 1:N+1
        if i == 1 || i == N+1
            w(i) = 1/(N^2);
        else
            theta_i = pi*(i-1)/N;
            w(i) = pi/N * sin(theta_i);
        end
    end
end

%% Simplified Chebyshev NMPC Optimization
function [U_opt, cost] = chebyshev_nmpc_optimize(x0, reference, taus, D, weights, T_scale, ...
                                               os_A, os_B, os_surge, u_min, u_max, r_min, r_max, ...
                                               w_position, w_orientation, w_control)
    N = length(taus) - 1;
    
    % Initial guess - zero control sequence
    U0 = zeros(N+1, 1);
    
    % Bounds
    lb = u_min * ones(N+1, 1);
    ub = u_max * ones(N+1, 1);
    
    % Optimization options
    options = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 100, ...
                          'Algorithm', 'sqp', 'StepTolerance', 1e-8);
    
    % Solve optimization
    [U_opt, cost] = fmincon(@(U) objective_function(U, x0, reference, taus, D, weights, T_scale, ...
                          os_A, os_B, os_surge, w_position, w_orientation, w_control), ...
                          U0, [], [], [], [], lb, ub, [], options);
end

%% Objective Function
function J = objective_function(U, x0, reference, taus, D, weights, T_scale, ...
                              os_A, os_B, os_surge, w_position, w_orientation, w_control)
    N = length(taus) - 1;
    
    % Solve for states using Chebyshev collocation
    X = solve_chebyshev_states(x0, U, taus, D, T_scale, os_A, os_B, os_surge);
    
    % Calculate objective
    J = 0;
    for i = 1:N+1
        % Position error
        pos_error = X(3:4, i) - reference(1:2);
        
        % Orientation error using atan2 for correct angle difference
        orientation_diff = atan2(sin(X(5, i) - reference(3)), cos(X(5, i) - reference(3)));
        
        % Control cost
        control_cost = U(i)^2;
        
        % Weighted sum
        J = J + weights(i) * (w_position * (pos_error' * pos_error) + ...
                             w_orientation * orientation_diff^2 + ...
                             w_control * control_cost);
    end
    J = T_scale * J;
end

%% Simplified State Solver
function X = solve_chebyshev_states(x0, U, taus, D, T_scale, os_A, os_B, os_surge)
    N = length(taus) - 1;
    n_states = 5;
    
    % Use forward simulation as approximation for initial guess
    X = zeros(n_states, N+1);
    X(:, 1) = x0;
    
    dt = T_scale * 2 / N; % Approximate time step
    
    for i = 2:N+1
        % Simple forward Euler integration
        x_dot = ship_dynamics(X(:, i-1), U(i-1), os_A, os_B, os_surge);
        X(:, i) = X(:, i-1) + x_dot * dt;
    end
    
    % Refine using Chebyshev collocation (single iteration)
    for iter = 1:3 % Few iterations for refinement
        for i = 2:N+1
            % Chebyshev derivative approximation
            cheb_deriv = (2/T_scale) * D(i, :) * X';
            
            % Dynamics constraint
            x_dot = ship_dynamics(X(:, i), U(i), os_A, os_B, os_surge);
            
            % Update state using pseudo-correction
            correction = 0.1 * (cheb_deriv' - x_dot);
            X(:, i) = X(:, i) - correction;
        end
    end
end

%% Main MPC Loop
fprintf('Starting Chebyshev NMPC...\n');

% Storage arrays
array_state = x0;
control_sequence = [];
mpciter = 0;
current_state = x0;

% Distance condition
distance_condition = 10;

main_loop = tic;
while mpciter < simulation_time/time_sampling
    fprintf('MPC Iteration: %d\n', mpciter);
    
    % Solve Chebyshev NMPC
    try
        [U_opt, cost] = chebyshev_nmpc_optimize(current_state, reference_pose, taus, D, weights, ...
                                              T_scale, os_A, os_B, os_surge, u_min, u_max, ...
                                              r_min, r_max, w_position, w_orientation, w_control);
        
        % Apply first control
        u_apply = U_opt(1);
    catch
        fprintf('Optimization failed, using zero control\n');
        u_apply = 0;
    end
    
    control_sequence = [control_sequence; u_apply];
    
    % Simulate system forward using ODE45
    [~, x_sim] = ode45(@(t,x) ship_dynamics(x, u_apply, os_A, os_B, os_surge), ...
                       [0, time_sampling], current_state);
    current_state = x_sim(end, :)';
    
    % Store state
    array_state = [array_state, current_state];
    
    % Check distance to destination
    distance_to_destination = norm(current_state(3:4) - reference_pose(1:2));
    fprintf('Distance to destination: %.2f m\n', distance_to_destination);
    
    if distance_to_destination < distance_condition
        fprintf('Destination reached!\n');
        break;
    end
    
    mpciter = mpciter + 1;
end

main_loop_time = toc(main_loop);
fprintf('Total simulation time: %.2f seconds\n', main_loop_time);
fprintf('Average MPC time: %.4f seconds\n', main_loop_time/(mpciter+1));

%% Plotting Results
plot_results(array_state, control_sequence, time_sampling, reference_pose, ...
             r_min, r_max, u_min, u_max, rrot_min, rrot_max);

fprintf('Chebyshev NMPC simulation completed successfully!\n');

%% Plotting Function
function plot_results(array_state, control_sequence, time_sampling, reference_pose, ...
                     r_min, r_max, u_min, u_max, rrot_min, rrot_max)
    
    time = (0:size(array_state,2)-1) * time_sampling;
    control_time = (0:length(control_sequence)-1) * time_sampling;
    
    % Convert to degrees for plotting
    control_sequence_deg = rad2deg(control_sequence);
    u_min_deg = rad2deg(u_min); u_max_deg = rad2deg(u_max);
    rrot_min_deg = rad2deg(rrot_min); rrot_max_deg = rad2deg(rrot_max);
    
    line_width = 1.5;
    fontsize_labels = 14;
    
    % Position plot
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2,3,1);
    plot(array_state(3,:), array_state(4,:), 'b-', 'LineWidth', line_width);
    hold on;
    plot(reference_pose(1), reference_pose(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    xlabel('X Position (m)'); ylabel('Y Position (m)');
    title('Ship Trajectory'); grid on;
    legend('Actual Path', 'Reference', 'Location', 'best');
    
    % Sway velocity
    subplot(2,3,2);
    plot(time, array_state(1,:), 'b-', 'LineWidth', line_width);
    xlabel('Time (s)'); ylabel('Sway Velocity (m/s)');
    title('Sway Velocity'); grid on;
    
    % Yaw rate
    subplot(2,3,3);
    plot(time, array_state(2,:), 'b-', 'LineWidth', line_width);
    hold on;
    plot(time, r_max*ones(size(time)), 'r--', 'LineWidth', line_width);
    plot(time, r_min*ones(size(time)), 'r--', 'LineWidth', line_width);
    xlabel('Time (s)'); ylabel('Yaw Rate (rad/s)');
    title('Yaw Rate'); grid on;
    legend('Actual', 'Constraints', 'Location', 'best');
    
    % Heading
    subplot(2,3,4);
    heading_deg = rad2deg(array_state(5,:));
    ref_heading_deg = rad2deg(reference_pose(3));
    plot(time, heading_deg, 'b-', 'LineWidth', line_width);
    hold on;
    plot(time, ref_heading_deg*ones(size(time)), 'r--', 'LineWidth', line_width);
    xlabel('Time (s)'); ylabel('Heading (deg)');
    title('Ship Heading'); grid on;
    legend('Actual', 'Reference', 'Location', 'best');
    
    % Rudder angle
    subplot(2,3,5);
    stairs(control_time, control_sequence_deg, 'k-', 'LineWidth', line_width);
    hold on;
    plot(control_time, u_max_deg*ones(size(control_time)), 'r--', 'LineWidth', line_width);
    plot(control_time, u_min_deg*ones(size(control_time)), 'r--', 'LineWidth', line_width);
    xlabel('Time (s)'); ylabel('Rudder Angle (deg)');
    title('Rudder Control'); grid on;
    legend('Actual', 'Constraints', 'Location', 'best');
    
    % Rudder rate
    subplot(2,3,6);
    if length(control_sequence_deg) > 1
        rudder_rate = diff(control_sequence_deg)/time_sampling;
        rudder_time = control_time(1:end-1) + time_sampling/2;
        stairs(rudder_time, rudder_rate, 'k-', 'LineWidth', line_width);
        hold on;
        plot(rudder_time, rrot_max_deg*ones(size(rudder_time)), 'r--', 'LineWidth', line_width);
        plot(rudder_time, rrot_min_deg*ones(size(rudder_time)), 'r--', 'LineWidth', line_width);
        xlabel('Time (s)'); ylabel('Rudder Rate (deg/s)');
        title('Rudder Rate'); grid on;
        legend('Actual', 'Constraints', 'Location', 'best');
    end
    
    % Save figure
    saveas(gcf, 'chebyshev_nmpc_results.png');
end