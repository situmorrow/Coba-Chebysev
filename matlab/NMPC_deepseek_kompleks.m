% point stabilization + Single shooting - MATLAB NMPC
clc
clear all
close all

%% Parameters
time_sampling = 1; % seconds
Np = 60; % prediction horizon
simulation_time = 500;

% input constraints
r_max = 0.0932; r_min = -r_max;
u_max = deg2rad(35); u_min = -u_max;
rrot_max = deg2rad(5); rrot_min = -rrot_max;

%% Corvette SIGMA class parameters
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

%% Hydrodynamics Coefficient
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
nd_u = 1;
nd_m = m/(0.5*rho*Lpp^3);
nd_xG = xG/Lpp;

%% Mathematical model matrices
M = [nd_m-Yvdot nd_m*nd_xG-Yrdot;
     nd_m*nd_xG-Nvdot Iz-Nrdot];
N = [-Yv nd_m*nd_u-Yr;
     -Nv nd_m*nd_xG*nd_u-Nr];

model_A = -M\N; 
a11 = model_A(1,1); a12 = model_A(1,2);
a21 = model_A(2,1); a22 = model_A(2,2);

b = [0.01;1];
b11 = b(1); b12 = b(2);
model_B = [b11; b12/Lpp];

% Operating speed matrices
os_U = os_surge;
os_A = os_U*[a11 a12*Lpp; a21/Lpp a22]/Lpp;
os_B = os_U^2*model_B/Lpp;

%% Ship Model Function
function xdot = shipModel(x, rudder_angle, os_A, os_B, os_surge)
    v = x(1);
    r = x(2);
    psi = x(5);
    
    % Dynamics
    x_dynamics = [v; r];
    xdot_dynamics = os_A * x_dynamics + os_B * rudder_angle;
    
    % Kinematics
    xdot = [xdot_dynamics(1);
            xdot_dynamics(2);
            os_surge * cos(psi) - v * sin(psi);
            os_surge * sin(psi) + v * cos(psi);
            r];
end

%% Simulation function
function x_next = simulateShip(x, u, dt, os_A, os_B, os_surge)
    % Runge-Kutta 4th order integration
    k1 = shipModel(x, u, os_A, os_B, os_surge);
    k2 = shipModel(x + 0.5*dt*k1, u, os_A, os_B, os_surge);
    k3 = shipModel(x + 0.5*dt*k2, u, os_A, os_B, os_surge);
    k4 = shipModel(x + dt*k3, u, os_A, os_B, os_surge);
    
    x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
    
    % Wrap heading to [0, 2*pi]
    x_next(5) = mod(x_next(5), 2*pi);
end

%% Cost Function for NMPC - IMPROVED VERSION
function [J, predicted_states] = costFunction(U, x0, reference_pose, Np, dt, os_A, os_B, os_surge)
    % REVISED WEIGHTS - lebih menekankan pada pencapaian tujuan
    w_position = 1e-2;    % Increased from 1e-4
    w_control = 1e-3;     % Reduced control penalty
    w_orientation = 0.01; % Reduced orientation weight
    w_velocity = 1e-4;    % Added velocity penalty
    
    J = 0;
    x = x0;
    predicted_states = zeros(5, Np+1);
    predicted_states(:,1) = x0;
    
    for k = 1:Np
        % Get control for this step
        u_k = U(k);
        
        % Simulate one step
        x = simulateShip(x, u_k, dt, os_A, os_B, os_surge);
        predicted_states(:,k+1) = x;
        
        % Position error - utama
        pos_error = x(3:4) - reference_pose(1:2);
        
        % LOS heading calculation - improved
        dx = reference_pose(1) - x(3);
        dy = reference_pose(2) - x(4);
        desired_heading = atan2(dy, dx);
        
        % Normalize angle difference
        heading_error = atan2(sin(x(5) - desired_heading), cos(x(5) - desired_heading));
        
        % Distance to target - tambahkan ini untuk memperkuat konvergensi
        distance_to_target = norm(pos_error);
        
        % Accumulate cost - REVISED dengan penekanan berbeda
        J = J + w_position * (pos_error' * pos_error) + ...
                w_orientation * (heading_error^2) + ...
                w_control * (u_k^2) + ...
                1e-5 * distance_to_target; % Extra push toward target
    end
    
    % Terminal cost - lebih ditekankan
    pos_error_N = x(3:4) - reference_pose(1:2);
    dx_N = reference_pose(1) - x(3);
    dy_N = reference_pose(2) - x(4);
    desired_heading_N = atan2(dy_N, dx_N);
    heading_error_N = atan2(sin(x(5) - desired_heading_N), cos(x(5) - desired_heading_N));
    
    J = J + 10 * w_position * (pos_error_N' * pos_error_N) + ...
            5 * w_orientation * (heading_error_N^2);
end

%% Nonlinear Constraints Function
function [c, ceq] = nonlinearConstraints(U, x0, u_prev, Np, dt, r_min, r_max, rrot_min, rrot_max, os_A, os_B, os_surge)
    ceq = [];
    c = [];
    
    x = x0;
    
    % Yaw rate constraints
    for k = 1:Np
        u_k = U(k);
        x = simulateShip(x, u_k, dt, os_A, os_B, os_surge);
        c = [c; x(2) - r_max; r_min - x(2)]; % r_min <= yaw_rate <= r_max
    end
    
    % Rudder rate constraints
    for k = 1:min(length(U), Np)
        if k == 1
            rudder_rate = (U(k) - u_prev) / dt;
        else
            rudder_rate = (U(k) - U(k-1)) / dt;
        end
        c = [c; rudder_rate - rrot_max; rrot_min - rudder_rate];
    end
end

%% MPC Initialization
t0 = 0;
x0 = [0; 0; 0; 0; pi/2]; % initial condition
reference_pose = [500; 2000; 0]; % Reference posture
situation = 'improved_collision_free_matlab';
axis_lim = [-200 3600 -200 2200];

% Storage arrays
array_state = zeros(5, simulation_time + 1);
array_state(:, 1) = x0;
control_sequence = [];
u_prev = 0;

% MPC parameters
distance_condition = 15; % Reduced for earlier termination
mpciter = 0;

% Store prediction history for analysis
prediction_history = [];

fprintf('Starting IMPROVED MATLAB NMPC...\n');
fprintf('Target: [%.1f, %.1f]\n', reference_pose(1), reference_pose(2));

%% MPC Loop
main_loop = tic;
while mpciter < simulation_time/time_sampling
    current_x = array_state(:, mpciter + 1);
    
    % Check distance to destination
    distance_to_destination = norm(current_x(3:4) - reference_pose(1:2));
    if distance_to_destination <= distance_condition
        fprintf('Destination reached at iteration %d\n', mpciter);
        break;
    end
    
    % IMPROVED: Better initial guess - point toward target
    dx = reference_pose(1) - current_x(3);
    dy = reference_pose(2) - current_x(4);
    target_angle = atan2(dy, dx);
    heading_error = atan2(sin(target_angle - current_x(5)), cos(target_angle - current_x(5)));
    
    % Smart initial guess based on heading error
    if abs(heading_error) > pi/4
        U0 = sign(heading_error) * 0.2 * ones(Np, 1); % Turn toward target
    else
        U0 = zeros(Np, 1); % Small corrections
    end
    
    % Bounds for control
    lb = u_min * ones(Np, 1);
    ub = u_max * ones(Np, 1);
    
    % Optimization options - IMPROVED
    options = optimoptions('fmincon', ...
        'Display', 'off', ...
        'MaxIterations', 100, ...
        'Algorithm', 'sqp', ...
        'StepTolerance', 1e-6, ...
        'ConstraintTolerance', 1e-6);
    
    try
        % Solve NMPC problem with additional output
        [U_opt, fval, exitflag] = fmincon(@(U) costFunction(U, current_x, reference_pose, Np, time_sampling, os_A, os_B, os_surge), ...
            U0, [], [], [], [], lb, ub, ...
            @(U) nonlinearConstraints(U, current_x, u_prev, Np, time_sampling, r_min, r_max, rrot_min, rrot_max, os_A, os_B, os_surge), ...
            options);
        
        if exitflag <= 0
            fprintf('Warning: Optimization failed at iteration %d. Using fallback.\n', mpciter);
            % Fallback strategy
            U_opt = sign(heading_error) * 0.1 * ones(Np, 1);
        end
        
        % Apply first control
        u_opt = U_opt(1);
        control_sequence = [control_sequence; u_opt];
        
        % Simulate system with optimal control
        next_state = simulateShip(current_x, u_opt, time_sampling, os_A, os_B, os_surge);
        
        % Store results
        array_state(:, mpciter + 2) = next_state;
        u_prev = u_opt;
        
        % Get prediction for analysis
        [~, pred_states] = costFunction(U_opt, current_x, reference_pose, Np, time_sampling, os_A, os_B, os_surge);
        prediction_history = [prediction_history, pred_states];
        
        % Display progress
        if mod(mpciter, 5) == 0
            fprintf('Iter: %3d, Dist: %6.1f m, HeadErr: %5.1f°, Ctrl: %6.2f°\n', ...
                mpciter, distance_to_destination, rad2deg(heading_error), rad2deg(u_opt));
        end
        
    catch ME
        fprintf('Optimization failed at iteration %d: %s\n', mpciter, ME.message);
        % Emergency fallback
        u_opt = sign(heading_error) * 0.15;
        control_sequence = [control_sequence; u_opt];
        next_state = simulateShip(current_x, u_opt, time_sampling, os_A, os_B, os_surge);
        array_state(:, mpciter + 2) = next_state;
        u_prev = u_opt;
    end
    
    mpciter = mpciter + 1;
end

main_loop_time = toc(main_loop);
fprintf('\nMPC completed in %.2f seconds\n', main_loop_time);
fprintf('Final distance to target: %.2f m\n', distance_to_destination);
fprintf('Average iteration time: %.4f seconds\n', main_loop_time/mpciter);

%% Enhanced Plotting Results
time = (0:mpciter) * time_sampling;
control_time = (0:length(control_sequence)-1) * time_sampling;

% Convert to degrees for plotting
array_heading = rad2deg(array_state(5, 1:mpciter+1));
control_sequence_deg = rad2deg(control_sequence);

figure('Position', [100, 100, 1400, 900]);

% Position trajectory - MAIN PLOT
subplot(2,3,1);
plot(array_state(3, 1:mpciter+1), array_state(4, 1:mpciter+1), 'b-', 'LineWidth', 3);
hold on;
plot(reference_pose(1), reference_pose(2), 'ro', 'MarkerSize', 12, 'LineWidth', 3);
plot(array_state(3, 1), array_state(4, 1), 'go', 'MarkerSize', 10, 'LineWidth', 2);

% Plot some predictions
if ~isempty(prediction_history)
    for k = 1:5:min(20, size(prediction_history, 3))
        pred = prediction_history(:,:,k);
        plot(pred(3,:), pred(4,:), 'm:', 'LineWidth', 1, 'Alpha', 0.3);
    end
end

xlabel('X Position (m)', 'FontSize', 12);
ylabel('Y Position (m)', 'FontSize', 12);
title('Ship Trajectory - Improved NMPC', 'FontSize', 14);
legend('Actual Path', 'Target', 'Start', 'Predicted', 'Location', 'best');
grid on;
axis equal;

% Heading with desired heading
subplot(2,3,2);
desired_headings = zeros(1, mpciter+1);
for k = 1:mpciter+1
    dx = reference_pose(1) - array_state(3,k);
    dy = reference_pose(2) - array_state(4,k);
    desired_headings(k) = rad2deg(atan2(dy, dx));
end
plot(time, array_heading, 'b-', 'LineWidth', 2);
hold on;
plot(time, desired_headings, 'r--', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Heading (deg)', 'FontSize', 12);
title('Ship Heading vs Desired', 'FontSize', 14);
legend('Actual Heading', 'Desired Heading', 'Location', 'best');
grid on;

% Distance to target
subplot(2,3,3);
distances = zeros(1, mpciter+1);
for k = 1:mpciter+1
    distances(k) = norm(array_state(3:4,k) - reference_pose(1:2));
end
plot(time, distances, 'g-', 'LineWidth', 2);
hold on;
plot([time(1), time(end)], [distance_condition, distance_condition], 'r--', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Distance to Target (m)', 'FontSize', 12);
title('Convergence to Target', 'FontSize', 14);
legend('Distance', 'Target Radius', 'Location', 'best');
grid on;

% Yaw rate
subplot(2,3,4);
plot(time, array_state(2, 1:mpciter+1), 'b-', 'LineWidth', 2);
hold on;
plot([time(1), time(end)], [r_max, r_max], 'r--', 'LineWidth', 1.5);
plot([time(1), time(end)], [r_min, r_min], 'r--', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Yaw Rate (rad/s)', 'FontSize', 12);
title('Yaw Rate', 'FontSize', 14);
legend('Actual', 'Constraints', 'Location', 'best');
grid on;

% Rudder angle
subplot(2,3,5);
stairs(control_time, control_sequence_deg, 'b-', 'LineWidth', 2);
hold on;
plot([control_time(1), control_time(end)], [rad2deg(u_max), rad2deg(u_max)], 'r--', 'LineWidth', 1.5);
plot([control_time(1), control_time(end)], [rad2deg(u_min), rad2deg(u_min)], 'r--', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Rudder Angle (deg)', 'FontSize', 12);
title('Rudder Control', 'FontSize', 14);
legend('Actual', 'Constraints', 'Location', 'best');
grid on;

% Progress monitoring
subplot(2,3,6);
progress = (1 - distances./distances(1)) * 100;
plot(time, progress, 'k-', 'LineWidth', 3);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Mission Completion (%)', 'FontSize', 12);
title('Mission Progress', 'FontSize', 14);
grid on;
ylim([0 100]);

sgtitle('IMPROVED MATLAB NMPC - Ship Control Results', 'FontSize', 16);

%% Save results
results = struct();
results.time = time;
results.states = array_state(:, 1:mpciter+1);
results.controls = control_sequence;
results.reference = reference_pose;
results.computation_time = main_loop_time;
results.final_distance = distance_to_destination;

save([situation '_results.mat'], 'results');
fprintf('Results saved to %s_results.mat\n', situation);

%% Display summary
fprintf('\n=== MISSION SUMMARY ===\n');
fprintf('Start position: [%.1f, %.1f]\n', array_state(3,1), array_state(4,1));
fprintf('Final position: [%.1f, %.1f]\n', array_state(3,end), array_state(4,end));
fprintf('Target position: [%.1f, %.1f]\n', reference_pose(1), reference_pose(2));
fprintf('Total travel distance: %.1f m\n', sum(sqrt(diff(array_state(3,:)).^2 + diff(array_state(4,:)).^2)));
% fprintf('Mission success: %s\n', distance_to_destination <= distance_condition ? "YES" : "PARTIAL");