% Chebyshev Pseudospectral Method for NMPC - Point Stabilization
clc
clear all
close all

%% Parameters
time_sampling = 1; % seconds
N = 10; % Chebyshev polynomial order (collocation points = N+1)
T = 60; % prediction horizon time (seconds)

% Ship constraints
r_max = 0.0932; r_min = -r_max;
u_max = deg2rad(35); u_min = -u_max;
rrot_max = deg2rad(5); rrot_min = -rrot_max;

%% Corvette SIGMA class parameters
Lpp = 101.07; 
B = 14; 
T = 3.7; 
m = 2423 * 1e3; 
os_surge = 15.4; % m/s
CB = 0.65; 
xG = 5.25; 
rho = 1024; 
Adelta = 5.7224; 
gyration = 0.156*Lpp; 

%% Hydrodynamics Coefficients
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

nd_u = 1;
nd_m = m/(0.5*rho*Lpp^3);
nd_xG = xG/Lpp;

%% Mathematical model coefficients
M = [nd_m-Yvdot nd_m*nd_xG-Yrdot;
     nd_m*nd_xG-Nvdot Iz-Nrdot];
N_mat = [-Yv nd_m*nd_u-Yr;
         -Nv nd_m*nd_xG*nd_u-Nr];

model_A = -inv(M)*N_mat; 
a11 = model_A(1,1); a12 = model_A(1,2);
a21 = model_A(2,1); a22 = model_A(2,2);

b = [0.01; 1];
b11 = b(1); b12 = b(2);
model_B = [b11; b12/Lpp];

os_U = os_surge;
os_A = os_U*[a11 a12*Lpp; a21/Lpp a22]/Lpp;
os_B = os_U^2*model_B/Lpp;

%% Chebyshev Setup
% Generate CGL nodes
tau = zeros(N+1, 1);
for k = 0:N
    tau(k+1) = cos(pi*k/N);
end

% Compute Chebyshev differentiation matrix
D = chebyshev_diff_matrix(N, tau);

% Compute CGL quadrature weights
w = chebyshev_weights(N);

%% Initial conditions and reference
x0 = [0; 0; 0; 0; pi/2]; % [v, r, x, y, psi]
reference_pose = [500; 2000; 0]; % [x_ref, y_ref, psi_ref]
situation = 'chebyshev_collision_free';
axis_lim = [-200 3600 -200 2200];

%% Objective weights
w_position = 1e-4;
w_control = 1e-2;
w_orientation = 0.1;

%% MPC Loop Setup
simulation_time = 300;
mpciter = 0;
array_state = x0;
control_sequence = [0];
distance_condition = 10;

main_loop = tic;
current_state = x0;

while mpciter < simulation_time/time_sampling
    distance_to_destination = norm(current_state(3:4) - reference_pose(1:2));
    
    if distance_to_destination < distance_condition
        break;
    end
    
    fprintf('MPC Iteration: %d, Distance: %.2f m\n', mpciter, distance_to_destination);
    
    % Solve Chebyshev NMPC
    [S_opt, U_opt, exitflag] = solve_chebyshev_nmpc(current_state, reference_pose, ...
        N, T, D, w, tau, os_surge, os_A, os_B, ...
        w_position, w_orientation, w_control, ...
        r_min, r_max, u_min, u_max, rrot_min, rrot_max);
    
    if exitflag <= 0
        warning('Optimization failed at iteration %d', mpciter);
    end
    
    % Apply first control input
    u_current = U_opt(1);
    control_sequence = [control_sequence; u_current];
    
    % Simulate system forward
    next_state = simulate_ship_dynamics(current_state, u_current, ...
        time_sampling, os_surge, os_A, os_B);
    
    current_state = next_state;
    array_state = [array_state, current_state];
    
    mpciter = mpciter + 1;
end

main_loop_time = toc(main_loop);
average_mpc_time = main_loop_time / (mpciter + 1);

fprintf('\n=== Simulation Complete ===\n');
fprintf('Final distance to destination: %.2f m\n', distance_to_destination);
fprintf('Average MPC computation time: %.4f s\n', average_mpc_time);
fprintf('Total iterations: %d\n', mpciter);

%% Plotting
total_time_index = size(array_state, 2);
time_vec = (0:total_time_index-1) * time_sampling;

plot_results(array_state, control_sequence, time_vec, reference_pose, ...
    r_min, r_max, u_min, u_max, rrot_min, rrot_max, situation);

%% Save data
csv_filename = strcat(situation, '_state_data.csv');
csv_data = [time_vec; array_state];
writematrix(csv_data, csv_filename);

%% Functions

function D = chebyshev_diff_matrix(N, tau)
    % Chebyshev-Gauss-Lobatto differentiation matrix
    D = zeros(N+1, N+1);
    c = ones(N+1, 1);
    c(1) = 2; c(end) = 2;
    
    for k = 0:N
        for i = 0:N
            if k ~= i
                D(k+1, i+1) = (c(k+1)/c(i+1)) * ((-1)^(k+i)) / (tau(k+1) - tau(i+1));
            end
        end
    end
    
    % Diagonal elements
    for k = 0:N
        if k == 0
            D(k+1, k+1) = (2*N^2 + 1) / 6;
        elseif k == N
            D(k+1, k+1) = -(2*N^2 + 1) / 6;
        else
            D(k+1, k+1) = -tau(k+1) / (2*(1 - tau(k+1)^2));
        end
    end
end

function w = chebyshev_weights(N)
    % CGL quadrature weights
    w = zeros(N+1, 1);
    
    if mod(N, 2) == 0  % N even
        w(1) = 1 / (N^2 - 1);
        w(N+1) = w(1);
        
        for k = 1:N-1
            sum_gamma = 0;
            for i = 0:N/2
                if i == 0 || i == N/2
                    gamma_val = 1 / (1 - 4*i^2) * cos(2*pi*i*k/N);
                else
                    gamma_val = 2 / (1 - 4*i^2) * cos(2*pi*i*k/N);
                end
                sum_gamma = sum_gamma + gamma_val;
            end
            w(k+1) = 2/N * sum_gamma;
        end
    else  % N odd
        w(1) = 1 / N^2;
        w(N+1) = w(1);
        
        for k = 1:N-1
            sum_gamma = 0;
            for i = 0:(N-1)/2
                if i == 0
                    gamma_val = 1 / (1 - 4*i^2) * cos(2*pi*i*k/N);
                else
                    gamma_val = 2 / (1 - 4*i^2) * cos(2*pi*i*k/N);
                end
                sum_gamma = sum_gamma + gamma_val;
            end
            w(k+1) = 2/N * sum_gamma;
        end
    end
end

function f = ship_dynamics(state, control, os_surge, os_A, os_B)
    % Nonlinear ship dynamics
    v = state(1);
    r = state(2);
    psi = state(5);
    
    dynamics_vr = os_A * [v; r] + os_B * control;
    
    f = [dynamics_vr(1);
         dynamics_vr(2);
         os_surge*cos(psi) - v*sin(psi);
         os_surge*sin(psi) + v*cos(psi);
         r];
end

function next_state = simulate_ship_dynamics(state, control, dt, os_surge, os_A, os_B)
    % Simple Euler integration
    f = ship_dynamics(state, control, os_surge, os_A, os_B);
    next_state = state + dt * f;
    next_state(5) = mod(next_state(5), 2*pi);
end

function [S_opt, U_opt, exitflag] = solve_chebyshev_nmpc(x0, ref_pose, ...
    N, T, D, w, tau, os_surge, os_A, os_B, ...
    w_pos, w_ori, w_ctrl, r_min, r_max, u_min, u_max, rrot_min, rrot_max)
    
    % Number of states and controls at each collocation point
    n_states = 5;
    n_controls = 1;
    n_points = N + 1;
    
    % Decision variables: [S0, S1, ..., SN, U0, U1, ..., UN]
    % where Si = [vi, ri, xi, yi, psii]
    n_vars = n_points * (n_states + n_controls);
    
    % Initial guess (straight line trajectory)
    S_init = repmat(x0, 1, n_points);
    for i = 1:n_points
        alpha = (i-1) / N;
        S_init(3:4, i) = x0(3:4) + alpha * (ref_pose(1:2) - x0(3:4));
        S_init(5, i) = x0(5) + alpha * (ref_pose(3) - x0(5));
    end
    U_init = zeros(n_controls, n_points);
    
    X0 = [reshape(S_init, [], 1); reshape(U_init, [], 1)];
    
    % Bounds
    lb = -inf(n_vars, 1);
    ub = inf(n_vars, 1);
    
    % State bounds
    for i = 1:n_points
        idx_base = (i-1) * n_states;
        % v bounds (sway velocity)
        lb(idx_base + 1) = -10;
        ub(idx_base + 1) = 10;
        % r bounds (yaw rate)
        lb(idx_base + 2) = r_min;
        ub(idx_base + 2) = r_max;
    end
    
    % Control bounds
    idx_ctrl_start = n_points * n_states;
    for i = 1:n_points
        lb(idx_ctrl_start + i) = u_min;
        ub(idx_ctrl_start + i) = u_max;
    end
    
    % Setup optimization
    options = optimoptions('fmincon', ...
        'Display', 'off', ...
        'Algorithm', 'sqp', ...
        'MaxIterations', 500, ...
        'MaxFunctionEvaluations', 10000, ...
        'ConstraintTolerance', 1e-6, ...
        'OptimalityTolerance', 1e-6);
    
    % Solve
    [X_opt, ~, exitflag] = fmincon(@(X) objective_function(X, N, n_states, n_controls, ...
        ref_pose, w, w_pos, w_ori, w_ctrl, T, D), ...
        X0, [], [], [], [], lb, ub, ...
        @(X) nonlinear_constraints(X, N, n_states, n_controls, x0, ...
        D, T, os_surge, os_A, os_B, rrot_min, rrot_max), ...
        options);
    
    % Extract solution
    S_opt = reshape(X_opt(1:n_points*n_states), n_states, n_points);
    U_opt = X_opt(n_points*n_states+1:end);
end

function J = objective_function(X, N, n_states, n_controls, ref_pose, w, ...
    w_pos, w_ori, w_ctrl, T, D)
    
    n_points = N + 1;
    
    % Extract states and controls
    S = reshape(X(1:n_points*n_states), n_states, n_points);
    U = X(n_points*n_states+1:end);
    
    % Compute cost at each collocation point
    J = 0;
    for k = 1:n_points
        % Position error
        pos_error = S(3:4, k) - ref_pose(1:2);
        
        % Orientation error (shortest angle)
        psi_ship = mod(S(5, k), 2*pi);
        psi_ref = mod(ref_pose(3), 2*pi);
        ori_diff = psi_ship - psi_ref;
        ori_diff = mod(ori_diff + pi, 2*pi) - pi;
        ori_error = abs(ori_diff);
        
        % Control effort
        ctrl_cost = U(k)^2;
        
        % Control rate (approximated using differentiation matrix)
        if k < n_points
            du = 0;
            for i = 1:n_points
                du = du + D(k, i) * U(i);
            end
            du = du * 2/T; % scale by time transformation
            ctrl_rate_cost = du^2;
        else
            ctrl_rate_cost = 0;
        end
        
        % Weighted sum with quadrature weight
        J = J + w(k) * (w_pos * norm(pos_error)^2 + ...
                        w_ori * ori_error^2 + ...
                        w_ctrl * (ctrl_cost + ctrl_rate_cost));
    end
    
    % Scale by time transformation
    J = J * T / 2;
end

function [c, ceq] = nonlinear_constraints(X, N, n_states, n_controls, x0, ...
    D, T, os_surge, os_A, os_B, rrot_min, rrot_max)
    
    n_points = N + 1;
    
    % Extract states and controls
    S = reshape(X(1:n_points*n_states), n_states, n_points);
    U = X(n_points*n_states+1:end);
    
    % Collocation constraints: D*S = (T/2)*f(S, U)
    ceq = [];
    
    for k = 1:n_points
        % Compute derivative at collocation point k
        dS_k = zeros(n_states, 1);
        for i = 1:n_points
            dS_k = dS_k + D(k, i) * S(:, i);
        end
        
        % Compute dynamics at collocation point k
        f_k = ship_dynamics(S(:, k), U(k), os_surge, os_A, os_B);
        
        % Collocation equation: dS_k = (T/2) * f_k
        ceq = [ceq; dS_k - (T/2) * f_k];
    end
    
    % Initial condition constraint
    ceq = [ceq; S(:, 1) - x0];
    
    % Control rate constraints (inequality)
    c = [];
    for k = 1:N
        du = U(k+1) - U(k);
        c = [c; du - rrot_max; -du - rrot_max];
    end
end

function plot_results(array_state, control_sequence, time_vec, reference_pose, ...
    r_min, r_max, u_min, u_max, rrot_min, rrot_max, situation)
    
    line_width = 1.5;
    fontsize_labels = 14;
    
    % Sway velocity
    figure('Name', 'Sway Velocity');
    plot(time_vec, array_state(1,:), '-b', 'LineWidth', line_width);
    ylabel('Sway velocity (m/s)', 'Interpreter', 'latex', 'FontSize', fontsize_labels);
    xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', fontsize_labels);
    grid on;
    saveas(gcf, [situation '_sway.png']);
    
    % Yaw rate
    figure('Name', 'Yaw Rate');
    plot(time_vec, array_state(2,:), '-b', 'LineWidth', line_width);
    hold on;
    plot(time_vec, r_max*ones(size(time_vec)), '-r', 'LineWidth', line_width);
    plot(time_vec, r_min*ones(size(time_vec)), '-r', 'LineWidth', line_width);
    ylabel('Yaw rate (rad/s)', 'Interpreter', 'latex', 'FontSize', fontsize_labels);
    xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', fontsize_labels);
    legend('Ship yaw rate', 'Constraints');
    grid on;
    saveas(gcf, [situation '_yaw.png']);
    
    % Heading
    figure('Name', 'Heading');
    plot(time_vec, rad2deg(array_state(5,:)), '-b', 'LineWidth', line_width);
    hold on;
    plot(time_vec, rad2deg(reference_pose(3))*ones(size(time_vec)), '--r', 'LineWidth', line_width);
    ylabel('Heading (degree)', 'Interpreter', 'latex', 'FontSize', fontsize_labels);
    xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', fontsize_labels);
    legend('Ship heading', 'Reference');
    grid on;
    saveas(gcf, [situation '_heading.png']);
    
    % Rudder angle
    figure('Name', 'Rudder Angle');
    stairs(time_vec(1:end-1), rad2deg(control_sequence(2:end)), 'k', 'LineWidth', line_width);
    hold on;
    stairs(time_vec(1:end-1), rad2deg(u_max)*ones(size(time_vec(1:end-1))), '-r', 'LineWidth', line_width);
    stairs(time_vec(1:end-1), rad2deg(u_min)*ones(size(time_vec(1:end-1))), '-r', 'LineWidth', line_width);
    ylabel('Rudder angle (degree)');
    xlabel('Time (s)');
    legend('Rudder angle', 'Constraints');
    grid on;
    saveas(gcf, [situation '_rudder.png']);
    
    % Trajectory
    figure('Name', 'Ship Trajectory');
    plot(array_state(3,:), array_state(4,:), '-b', 'LineWidth', line_width);
    hold on;
    plot(reference_pose(1), reference_pose(2), 'rx', 'MarkerSize', 15, 'LineWidth', 2);
    ylabel('$y_E$-position (m)', 'Interpreter', 'latex', 'FontSize', fontsize_labels);
    xlabel('$x_E$-position (m)', 'Interpreter', 'latex', 'FontSize', fontsize_labels);
    legend('Ship trajectory', 'Target');
    grid on;
    axis equal;
    saveas(gcf, [situation '_trajectory.png']);
end