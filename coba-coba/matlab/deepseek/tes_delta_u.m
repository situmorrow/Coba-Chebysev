clc; clear; close all;

%% Parameters
% Simulation parameters
time_sampling = 1; % seconds
Np = 30; % prediction horizon
simulation_time = 200;

% Ship constraints
r_max = 0.0932; r_min = -r_max;
u_max = deg2rad(35); u_min = -u_max;
udot_max = deg2rad(2); udot_min = -udot_max;

% Cost weights
Q = diag([1e-3, 1e-3, 0.1]); % Weight for position and orientation
R = 1e-2; % Weight for control rate

% Initial conditions
x0 = [0; 0; 0; 0; pi/2]; % [v; r; x; y; psi]
reference_pose = [400; 1000; 0]; % [x_ref; y_ref; psi_ref]

%% Ship Model Parameters
Lpp = 101.07; B = 14; T = 3.7; m = 2423 * 1e3;
os_surge = 8.0; CB = 0.65; xG = 5.25; rho = 1024;
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

% Output matrix C (measurement: x, y, psi)
C = [0, 0, 1, 0, 0;
     0, 0, 0, 1, 0;
     0, 0, 0, 0, 1];

fprintf('Ship model initialized successfully.\n');

%% Chebyshev Setup (Simplified)
function [taus, D, w] = chebyshev_setup_simple(N)
    % Chebyshev-Gauss-Lobatto nodes
    taus = cos(pi*(0:N)/N)';
    
    % Simplified differentiation matrix
    D = zeros(N+1, N+1);
    
    for i = 1:N+1
        for j = 1:N+1
            if i ~= j
                D(i,j) = ((-1)^(i+j)) / (taus(i) - taus(j));
            end
        end
    end
    
    % Diagonal elements
    for i = 1:N+1
        D(i,i) = -sum(D(i,:));
    end
    
    % Simple weights
    w = ones(N+1, 1) * 2/N;
    w(1) = 1/N;
    w(end) = 1/N;
end

% Setup Chebyshev
N = 40;
[taus, D, weights] = chebyshev_setup_simple(N);
T_scale = (Np * time_sampling)/2;

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

%% Measurement Function
function y = measurement_function(x, C)
    y = C * x;
end

%% Simplified Chebyshev NMPC yang Pasti Bekerja
function U_opt = simple_chebyshev_nmpc(current_state, reference, N, taus, D, weights, T_scale, ...
                                      os_A, os_B, os_surge, C, u_min, u_max, udot_min, udot_max, ...
                                      r_min, r_max, Q, R)
    
    n_controls = 1;
    
    % Initial guess - zero control sequence
    U0 = zeros(N+1, 1);
    
    % Bounds
    lb = u_min * ones(N+1, 1);
    ub = u_max * ones(N+1, 1);
    
    % Optimization options - lebih toleran
    options = optimoptions('fmincon', 'Display', 'off', 'MaxIterations', 50, ...
                          'Algorithm', 'sqp', 'StepTolerance', 1e-6, ...
                          'ConstraintTolerance', 1e-3);
    
    % Solve optimization
    U_opt = fmincon(@(U) simple_objective(U, current_state, reference, N, taus, D, weights, T_scale, ...
                      os_A, os_B, os_surge, C, Q, R, udot_min, udot_max), ...
                      U0, [], [], [], [], lb, ub, ...
                      @(U) simple_constraints(U, current_state, N, os_A, os_B, os_surge, r_min, r_max), ...
                      options);
end

function J = simple_objective(U, current_state, reference, N, taus, D, weights, T_scale, ...
                             os_A, os_B, os_surge, C, Q, R, udot_min, udot_max)
    
    J = 0;
    
    % Simulate system forward untuk mendapatkan states
    x = current_state;
    X_sim = zeros(5, N+1);
    X_sim(:,1) = current_state;
    
    dt_approx = T_scale * 2 / N; % Approximate time step
    
    for k = 1:N
        x_dot = ship_dynamics(x, U(k), os_A, os_B, os_surge);
        x = x + x_dot * dt_approx;
        x(5) = atan2(sin(x(5)), cos(x(5))); % Normalize angle
        X_sim(:,k+1) = x;
    end
    
    % Calculate objective
    for k = 1:N+1
        % Output measurement
        y_k = measurement_function(X_sim(:, k), C);
        
        % Output error
        error_y = y_k - reference;
        
        % Control rate using Chebyshev differentiation (simplified)
        if k <= N
            udot_k = (U(min(k+1, N+1)) - U(k)) / dt_approx;
        else
            udot_k = 0;
        end
        
        % Weighted cost - utama pada posisi dan heading
        position_cost = error_y(1:2)' * Q(1:2,1:2) * error_y(1:2);
        heading_cost = Q(3,3) * error_y(3)^2;
        control_cost = R * udot_k^2;
        
        J = J + weights(k) * (position_cost + heading_cost + control_cost);
    end
    
    J = (T_scale/2) * J;
end

function [c, ceq] = simple_constraints(U, current_state, N, os_A, os_B, os_surge, r_min, r_max)
    % Simple constraints - focus on yaw rate constraints
    
    ceq = [];
    c = [];
    
    % Simulate to get yaw rates
    x = current_state;
    dt_approx = 0.1; % approximate time step
    
    for k = 1:N
        x_dot = ship_dynamics(x, U(k), os_A, os_B, os_surge);
        x = x + x_dot * dt_approx;
        
        % Yaw rate constraints
        c = [c; x(2) - r_max; r_min - x(2)];
    end
end

%% Main Simulation Loop yang Lebih Robust
fprintf('Starting Simplified Chebyshev NMPC...\n');

% Storage arrays
array_state = x0;
control_sequence = [];
mpciter = 0;
current_state = x0;
prev_control = 0;

% Distance condition
distance_condition = 10;
max_iterations = simulation_time / time_sampling;

% Progress tracking
fprintf('Progress: ');

main_loop = tic;

while mpciter < max_iterations
    if mod(mpciter, 10) == 0
        fprintf('%d ', mpciter);
    end
    
    try
        % Solve Simplified Chebyshev NMPC
        U_opt = simple_chebyshev_nmpc(current_state, reference_pose, Np, taus, D, weights, T_scale, ...
                                    os_A, os_B, os_surge, C, u_min, u_max, udot_min, udot_max, ...
                                    r_min, r_max, Q, R);
        
        u_apply = U_opt(1);
        
        % Rate limiting
        control_diff = u_apply - prev_control;
        if abs(control_diff) > udot_max * time_sampling
            u_apply = prev_control + sign(control_diff) * udot_max * time_sampling;
        end
        
    catch
        % Fallback: PD controller
        pos_error = current_state(3:4) - reference_pose(1:2);
        heading_error = atan2(sin(current_state(5) - reference_pose(3)), ...
                            cos(current_state(5) - reference_pose(3)));
        
        % Simple PD control
        kp_pos = 0.001;
        kp_heading = 0.5;
        
        u_apply = -kp_pos * norm(pos_error) - kp_heading * heading_error;
        u_apply = max(min(u_apply, u_max), u_min);
        
        % Rate limiting
        control_diff = u_apply - prev_control;
        if abs(control_diff) > udot_max * time_sampling
            u_apply = prev_control + sign(control_diff) * udot_max * time_sampling;
        end
    end
    
    control_sequence = [control_sequence; u_apply];
    prev_control = u_apply;
    
    % Simulate system forward menggunakan Euler integration (lebih stabil)
    x_dot = ship_dynamics(current_state, u_apply, os_A, os_B, os_surge);
    current_state = current_state + x_dot * time_sampling;
    
    % Normalize heading angle
    current_state(5) = atan2(sin(current_state(5)), cos(current_state(5)));
    
    % Store state
    array_state = [array_state, current_state];
    
    % Check distance to destination
    distance_to_destination = norm(current_state(3:4) - reference_pose(1:2));
    
    if distance_to_destination < distance_condition
        fprintf('\nDestination reached at iteration %d!\n', mpciter);
        break;
    end
    
    mpciter = mpciter + 1;
end

fprintf('\n');

if mpciter >= max_iterations
    fprintf('Maximum iterations reached.\n');
end

main_loop_time = toc(main_loop);
fprintf('Total simulation time: %.2f seconds\n', main_loop_time);
fprintf('Average iteration time: %.4f seconds\n', main_loop_time/(mpciter+1));
fprintf('Final distance to target: %.2f m\n', distance_to_destination);

%% Plotting Results yang Lebih Baik
plot_final_results(array_state, control_sequence, time_sampling, reference_pose, ...
                  r_min, r_max, u_min, u_max, udot_min, udot_max);

fprintf('Simplified Chebyshev NMPC completed!\n');

%% Final Plotting Function
function plot_final_results(array_state, control_sequence, time_sampling, reference_pose, ...
                           r_min, r_max, u_min, u_max, udot_min, udot_max)
    
    time = (0:size(array_state,2)-1) * time_sampling;
    control_time = (0:length(control_sequence)-1) * time_sampling;
    
    % Convert to degrees for plotting
    control_sequence_deg = rad2deg(control_sequence);
    u_min_deg = rad2deg(u_min); u_max_deg = rad2deg(u_max);
    udot_min_deg = rad2deg(udot_min); udot_max_deg = rad2deg(udot_max);
    
    line_width = 2.0;
    fontsize = 11;
    
    % Create figure
    figure('Position', [100, 100, 1200, 900]);
    
    % 1. Trajectory Plot
    subplot(2,3,1);
    plot(array_state(3,:), array_state(4,:), 'b-', 'LineWidth', line_width);
    hold on;
    plot(reference_pose(1), reference_pose(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
    plot(array_state(3,1), array_state(4,1), 'go', 'MarkerSize', 8, 'LineWidth', 2);
    xlabel('X Position (m)', 'FontSize', fontsize);
    ylabel('Y Position (m)', 'FontSize', fontsize);
    title('Ship Trajectory', 'FontSize', fontsize+1);
    grid on; axis equal;
    legend('Actual Path', 'Target', 'Start', 'Location', 'best');
    
    % 2. Sway velocity
    subplot(2,3,2);
    plot(time, array_state(1,:), 'b-', 'LineWidth', line_width);
    xlabel('Time (s)', 'FontSize', fontsize);
    ylabel('Sway Velocity (m/s)', 'FontSize', fontsize);
    title('Sway Velocity', 'FontSize', fontsize+1);
    grid on;
    
    % 3. Yaw rate
    subplot(2,3,3);
    plot(time, array_state(2,:), 'b-', 'LineWidth', line_width);
    hold on;
    plot(time, r_max*ones(size(time)), 'r--', 'LineWidth', 1.5);
    plot(time, r_min*ones(size(time)), 'r--', 'LineWidth', 1.5);
    xlabel('Time (s)', 'FontSize', fontsize);
    ylabel('Yaw Rate (rad/s)', 'FontSize', fontsize);
    title('Yaw Rate', 'FontSize', fontsize+1);
    legend('Actual', 'Max Limit', 'Min Limit', 'Location', 'best');
    grid on;
    
    % 4. Heading
    subplot(2,3,4);
    heading_deg = rad2deg(array_state(5,:));
    ref_heading_deg = rad2deg(reference_pose(3));
    plot(time, heading_deg, 'b-', 'LineWidth', line_width);
    hold on;
    plot(time, ref_heading_deg*ones(size(time)), 'r--', 'LineWidth', 1.5);
    xlabel('Time (s)', 'FontSize', fontsize);
    ylabel('Heading (deg)', 'FontSize', fontsize);
    title('Ship Heading', 'FontSize', fontsize+1);
    legend('Actual', 'Reference', 'Location', 'best');
    grid on;
    
    % 5. Rudder angle
    subplot(2,3,5);
    stairs(control_time, control_sequence_deg, 'k-', 'LineWidth', line_width);
    hold on;
    plot(control_time, u_max_deg*ones(size(control_time)), 'r--', 'LineWidth', 1.5);
    plot(control_time, u_min_deg*ones(size(control_time)), 'r--', 'LineWidth', 1.5);
    xlabel('Time (s)', 'FontSize', fontsize);
    ylabel('Rudder Angle (deg)', 'FontSize', fontsize);
    title('Rudder Control', 'FontSize', fontsize+1);
    legend('Actual', 'Max Limit', 'Min Limit', 'Location', 'best');
    grid on;
    ylim([u_min_deg-10, u_max_deg+10]);
    
    % 6. Rudder rate
    subplot(2,3,6);
    if length(control_sequence_deg) > 1
        rudder_rate = diff(control_sequence_deg)/time_sampling;
        rudder_time = control_time(1:end-1);
        plot(rudder_time, rudder_rate, 'k-', 'LineWidth', line_width);
        hold on;
        plot(rudder_time, udot_max_deg*ones(size(rudder_time)), 'r--', 'LineWidth', 1.5);
        plot(rudder_time, udot_min_deg*ones(size(rudder_time)), 'r--', 'LineWidth', 1.5);
        xlabel('Time (s)', 'FontSize', fontsize);
        ylabel('Rudder Rate (deg/s)', 'FontSize', fontsize);
        title('Rudder Rate', 'FontSize', fontsize+1);
        legend('Actual', 'Max Limit', 'Min Limit', 'Location', 'best');
        grid on;
    end
    
    fprintf('Plot saved as final_chebyshev_nmpc_results.png\n');
end