%% Chebyshev Pseudospectral NMPC - Ship Point Stabilization
% Pure MATLAB implementation - DEBUGGED VERSION
clc
clear all
close all

%% Parameters
time_horizon = 30; % prediction time horizon in seconds
N = 40; % number of Chebyshev collocation points

% Ship constraints
r_max = 0.0932; r_min = -r_max;
u_max = deg2rad(35); u_min = -u_max;
rrot_max = deg2rad(5); rrot_min = -rrot_max;

%% Ship Parameters - Corvette SIGMA class
Lpp = 101.07; 
B = 14; 
T = 3.7; 
m = 2423 * 1e3; 
os_surge = 15.4; 
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

M = [nd_m-Yvdot nd_m*nd_xG-Yrdot;
     nd_m*nd_xG-Nvdot Iz-Nrdot];
Nn = [-Yv nd_m*nd_u-Yr;
      -Nv nd_m*nd_xG*nd_u-Nr];

model_A = -M\Nn; 
a11 = model_A(1,1); a12 = model_A(1,2);
a21 = model_A(2,1); a22 = model_A(2,2);

b = [0.01; 1];
model_B = [b(1); b(2)/Lpp];

os_U = os_surge;
os_A = os_U*[a11 a12*Lpp; a21/Lpp a22]/Lpp;
os_B = os_U^2*model_B/Lpp;

%% Chebyshev Pseudospectral Setup
tau = zeros(N+1, 1);
for i = 0:N
    tau(i+1) = cos(i*pi/N);
end
tau = flipud(tau);

D = chebyshev_differentiation_matrix(N);
w = chebyshev_weights(N);

%% MPC Initialization
t0 = 0;
x0 = [0; 0; 0; 0; pi/2];
reference_pose = [500; 2000; 0];

% Weights (ADJUSTED FOR BETTER PERFORMANCE)
w_position = 5e-2;
w_control = 1e-4;
w_orientation = 5e-2;
w_terminal = 100; % Strong terminal cost

time_sampling = 1;
simulation_time = 300;
distance_condition = 10;

array_state = x0;
control_sequence = [];
mpciter = 0;

%% MPC Loop
main_loop = tic;
distance_to_destination = norm(x0(3:4) - reference_pose(1:2));

while (distance_to_destination > distance_condition && mpciter < simulation_time/time_sampling)
    fprintf('\n=== MPC Iteration: %d ===\n', mpciter);
    fprintf('Current position: [%.1f, %.1f], Distance: %.2f m\n', ...
        x0(3), x0(4), distance_to_destination);
    
    % Solve Chebyshev NMPC
    [u_opt, X_opt] = solve_chebyshev_nmpc(x0, reference_pose, N, tau, D, w, ...
        time_horizon, os_A, os_B, os_surge, ...
        r_min, r_max, u_min, u_max, rrot_min, rrot_max, ...
        w_position, w_control, w_orientation, w_terminal);
    
    % Apply first control
    u_current = u_opt(1);
    control_sequence = [control_sequence; u_current];
    
    % Check control rate
    if mpciter > 0
        du_actual = (u_current - control_sequence(end-1)) / time_sampling;
        fprintf('Control rate: %.4f rad/s (limit: ±%.4f rad/s)\n', ...
            du_actual, rrot_max);
    end
    
    % Simulate system
    xdot = ship_dynamics(x0, u_current, os_A, os_B, os_surge);
    x_next = x0 + time_sampling * xdot;
    x_next(5) = mod(x_next(5), 2*pi);
    
    x0 = x_next;
    array_state = [array_state, x0];
    distance_to_destination = norm(x0(3:4) - reference_pose(1:2));
    
    mpciter = mpciter + 1;
end

main_loop_time = toc(main_loop);
average_mpc_time = main_loop_time / (mpciter + 1);

fprintf('\n=== Simulation Complete ===\n');
fprintf('Final distance: %.2f m\n', distance_to_destination);
fprintf('Average MPC time: %.4f s\n', average_mpc_time);
fprintf('Total iterations: %d\n', mpciter);

%% Plotting
time = 0:time_sampling:(mpciter*time_sampling);
plot_results(time, array_state, control_sequence, reference_pose, ...
    r_min, r_max, u_min, u_max, rrot_min, rrot_max);

%% ===== FUNCTIONS =====

function D = chebyshev_differentiation_matrix(N)
    D = zeros(N+1, N+1);
    tau = zeros(N+1, 1);
    for i = 0:N
        tau(i+1) = cos(i*pi/N);
    end
    tau = flipud(tau);
    
    c = ones(N+1, 1);
    c(1) = 2;
    c(end) = 2;
    
    for i = 0:N
        for j = 0:N
            if i == j
                if i == 0
                    D(i+1, j+1) = (2*N^2 + 1) / 6;
                elseif i == N
                    D(i+1, j+1) = -(2*N^2 + 1) / 6;
                else
                    D(i+1, j+1) = -tau(i+1) / (2*(1 - tau(i+1)^2));
                end
            else
                sign_factor = (-1)^(i+j);
                D(i+1, j+1) = (c(i+1) / c(j+1)) * sign_factor / (tau(i+1) - tau(j+1));
            end
        end
    end
end

function w = chebyshev_weights(N)
    w = zeros(N+1, 1);
    
    if mod(N, 2) == 0
        w(1) = 1 / (N^2 - 1);
        w(end) = 1 / (N^2 - 1);
        
        for k = 1:N-1
            sum_term = gamma_func(0, k, N) + gamma_func(N/2, k, N);
            for i = 1:(N/2-1)
                sum_term = sum_term + 2*gamma_func(i, k, N);
            end
            w(k+1) = (2/N) * sum_term;
        end
    else
        w(1) = 1 / N^2;
        w(end) = 1 / N^2;
        
        for k = 1:N-1
            sum_term = gamma_func(0, k, N);
            for i = 1:((N-1)/2)
                sum_term = sum_term + 2*gamma_func(i, k, N);
            end
            w(k+1) = (2/N) * sum_term;
        end
    end
end

function g = gamma_func(i, k, N)
    g = (1 / (1 - 4*i^2)) * cos(2*pi*i*k/N);
end

function xdot = ship_dynamics(x, u, os_A, os_B, os_surge)
    v = x(1);
    r = x(2);
    psi = x(5);
    
    dynamics_vr = os_A * [v; r] + os_B * u;
    
    xdot = [dynamics_vr(1);
            dynamics_vr(2);
            os_surge*cos(psi) - v*sin(psi);
            os_surge*sin(psi) + v*cos(psi);
            r];
end

function [u_opt, X_opt] = solve_chebyshev_nmpc(x0, ref, N, tau, D, w, ...
    T, os_A, os_B, os_surge, r_min, r_max, u_min, u_max, rrot_min, rrot_max, ...
    w_pos, w_ctrl, w_orient, w_term)
    
    n_states = 5;
    n_controls = 1;
    n_vars = n_states*(N+1) + n_controls*(N+1);
    
    % Better initial guess
    z0 = zeros(n_vars, 1);
    
    % Compute desired heading to target
    dx = ref(1) - x0(3);
    dy = ref(2) - x0(4);
    desired_heading = atan2(dy, dx);
    
    for i = 0:N
        alpha = i / N;
        
        % Interpolate states
        v_interp = 0; % sway velocity
        r_interp = 0; % yaw rate
        x_interp = x0(3) + alpha * dx;
        y_interp = x0(4) + alpha * dy;
        
        % Smooth heading transition
        psi_diff = angle_wrap(desired_heading - x0(5));
        psi_interp = x0(5) + alpha * psi_diff;
        psi_interp = mod(psi_interp, 2*pi);
        
        z0(i*n_states + (1:n_states)) = [v_interp; r_interp; x_interp; y_interp; psi_interp];
    end
    
    % Control guess: gentle turning
    target_rudder = atan2(ref(2) - x0(4), ref(1) - x0(3)) - x0(5);
    target_rudder = angle_wrap(target_rudder);
    target_rudder = max(min(target_rudder/5, u_max*0.5), u_min*0.5); % conservative
    z0(n_states*(N+1)+1:end) = target_rudder;
    
    % Bounds
    lb = -inf(n_vars, 1);
    ub = inf(n_vars, 1);
    
    % Control bounds
    for i = 0:N
        idx_u = n_states*(N+1) + i + 1;
        lb(idx_u) = u_min;
        ub(idx_u) = u_max;
    end
    
    % Yaw rate bounds
    for i = 0:N
        idx_r = i*n_states + 2;
        lb(idx_r) = r_min;
        ub(idx_r) = r_max;
    end
    
    % Objective and constraints
    obj_fun = @(z) objective_function(z, N, w, T, ref, w_pos, w_ctrl, w_orient, w_term);
    nonlcon = @(z) nonlinear_constraints(z, N, D, T, x0, os_A, os_B, os_surge, rrot_min, rrot_max);
    
    % Optimization options
    options = optimoptions('fmincon', ...
        'Display', 'off', ...
        'Algorithm', 'sqp', ...
        'MaxIterations', 500, ...
        'MaxFunctionEvaluations', 5000, ...
        'ConstraintTolerance', 1e-3, ...
        'OptimalityTolerance', 1e-3, ...
        'StepTolerance', 1e-6);
    
    % Solve
    [z_opt, fval, exitflag, output] = fmincon(obj_fun, z0, [], [], [], [], lb, ub, nonlcon, options);
    
    % Check violations
    [c_check, ceq_check] = nonlcon(z_opt);
    max_ineq_viol = 0;
    if ~isempty(c_check)
        max_ineq_viol = max([0; c_check]);
    end
    max_eq_viol = max(abs(ceq_check));
    
    fprintf('Cost: %.1f, Exit: %d, Ineq: %.2e, Eq: %.2e, Iter: %d\n', ...
        fval, exitflag, max_ineq_viol, max_eq_viol, output.iterations);
    
    % Extract solution
    X_opt = reshape(z_opt(1:n_states*(N+1)), n_states, N+1);
    u_opt = z_opt(n_states*(N+1)+1:end);
    
    % Debug: show predicted trajectory endpoint
    fprintf('Predicted endpoint: [%.1f, %.1f], heading: %.1f deg\n', ...
        X_opt(3,end), X_opt(4,end), rad2deg(X_opt(5,end)));
end

function J = objective_function(z, N, w, T, ref, w_pos, w_ctrl, w_orient, w_term)
    n_states = 5;
    
    X = reshape(z(1:n_states*(N+1)), n_states, N+1);
    U = z(n_states*(N+1)+1:end);
    
    J = 0;
    
    % Stage cost
    for i = 0:N
        x_ship = X(3, i+1);
        y_ship = X(4, i+1);
        psi = X(5, i+1);
        u = U(i+1);
        
        pos_error = [x_ship - ref(1); y_ship - ref(2)];
        heading_error = angle_wrap(psi - ref(3));
        
        stage_cost = w_pos * (pos_error' * pos_error) + ...
                     w_orient * heading_error^2 + ...
                     w_ctrl * u^2;
        
        J = J + w(i+1) * stage_cost;
    end
    
    J = (T/2) * J;
    
    % Terminal cost (very important!)
    x_final = X(3, end);
    y_final = X(4, end);
    psi_final = X(5, end);
    
    pos_error_final = [x_final - ref(1); y_final - ref(2)];
    heading_error_final = angle_wrap(psi_final - ref(3));
    
    J = J + w_term * (w_pos * (pos_error_final' * pos_error_final) + ...
                      w_orient * heading_error_final^2);
    
    % Control smoothness
    for i = 1:N
        du = U(i+1) - U(i);
        J = J + 1e-4 * du^2;
    end
end

function [c, ceq] = nonlinear_constraints(z, N, D, T, x0, os_A, os_B, os_surge, rrot_min, rrot_max)
    n_states = 5;
    
    X = reshape(z(1:n_states*(N+1)), n_states, N+1);
    U = z(n_states*(N+1)+1:end);
    
    % === EQUALITY: Initial condition + dynamics ===
    ceq = X(:,1) - x0;
    
    % Collocation equations
    for k = 1:N
        x_dot = zeros(n_states, 1);
        for j = 0:N
            x_dot = x_dot + D(k+1, j+1) * X(:, j+1);
        end
        x_dot = (2/T) * x_dot;
        
        f_k = ship_dynamics(X(:, k+1), U(k+1), os_A, os_B, os_surge);
        ceq = [ceq; x_dot - f_k];
    end
    
    % === INEQUALITY: Control rate ===
    c = [];
    
    % Method 1: Using finite differences (more robust)
    for k = 1:N
        du_dt = (U(k+1) - U(k)) / (T / N); % approximate rate
        
        % Upper bound
        c = [c; du_dt - rrot_max];
        % Lower bound
        c = [c; -du_dt - (-rrot_min)];
    end
end

function wrapped = angle_wrap(angle)
    wrapped = mod(angle + pi, 2*pi) - pi;
end

function plot_results(time, states, controls, ref, r_min, r_max, u_min, u_max, rrot_min, rrot_max)
    line_width = 1.5;
    fontsize_labels = 12;
    
    % Trajectory
    figure('Position', [100, 100, 800, 600]);
    plot(states(3,:), states(4,:), 'b-', 'LineWidth', 2);
    hold on;
    plot(states(3,1), states(4,1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    plot(ref(1), ref(2), 'rx', 'MarkerSize', 15, 'LineWidth', 3);
    grid on;
    xlabel('X Position (m)', 'FontSize', fontsize_labels);
    ylabel('Y Position (m)', 'FontSize', fontsize_labels);
    title('Ship Trajectory', 'FontSize', fontsize_labels+2);
    legend('Trajectory', 'Start', 'Target', 'Location', 'best');
    axis equal;
    
    % States
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(3,2,1);
    plot(time, states(1,:), 'b-', 'LineWidth', line_width);
    grid on;
    xlabel('Time (s)'); ylabel('Sway Velocity (m/s)');
    title('Sway Velocity');
    
    subplot(3,2,2);
    plot(time, states(2,:), 'b-', 'LineWidth', line_width);
    hold on;
    plot(time, r_max*ones(size(time)), 'r--', 'LineWidth', line_width);
    plot(time, r_min*ones(size(time)), 'r--', 'LineWidth', line_width);
    grid on;
    xlabel('Time (s)'); ylabel('Yaw Rate (rad/s)');
    title('Yaw Rate');
    legend('Actual', 'Limits');
    
    subplot(3,2,3);
    plot(time, states(3,:), 'b-', 'LineWidth', line_width);
    hold on;
    plot(time, ref(1)*ones(size(time)), 'r--', 'LineWidth', line_width);
    grid on;
    xlabel('Time (s)'); ylabel('X (m)');
    title('X Position');
    
    subplot(3,2,4);
    plot(time, states(4,:), 'b-', 'LineWidth', line_width);
    hold on;
    plot(time, ref(2)*ones(size(time)), 'r--', 'LineWidth', line_width);
    grid on;
    xlabel('Time (s)'); ylabel('Y (m)');
    title('Y Position');
    
    subplot(3,2,5);
    plot(time, rad2deg(states(5,:)), 'b-', 'LineWidth', line_width);
    hold on;
    plot(time, rad2deg(ref(3))*ones(size(time)), 'r--', 'LineWidth', line_width);
    grid on;
    xlabel('Time (s)'); ylabel('Heading (deg)');
    title('Heading');
    
    subplot(3,2,6);
    control_time = time(1:end-1);
    stairs(control_time, rad2deg(controls), 'k-', 'LineWidth', line_width);
    hold on;
    plot(control_time, rad2deg(u_max)*ones(size(control_time)), 'r--', 'LineWidth', line_width);
    plot(control_time, rad2deg(u_min)*ones(size(control_time)), 'r--', 'LineWidth', line_width);
    grid on;
    xlabel('Time (s)'); ylabel('Rudder (deg)');
    title('Rudder Angle');
    
    % Control rate
    figure('Position', [100, 100, 800, 400]);
    if length(controls) > 1
        control_rate = diff(rad2deg(controls));
        stairs(control_time(2:end), control_rate, 'k-', 'LineWidth', line_width);
        hold on;
        plot(control_time(2:end), rad2deg(rrot_max)*ones(length(control_time)-1, 1), 'r--', 'LineWidth', line_width);
        plot(control_time(2:end), rad2deg(rrot_min)*ones(length(control_time)-1, 1), 'r--', 'LineWidth', line_width);
        grid on;
        xlabel('Time (s)'); ylabel('Rudder Rate (deg/s)');
        title('Control Rate of Change');
        legend('Rate', 'Constraints');
    end
end