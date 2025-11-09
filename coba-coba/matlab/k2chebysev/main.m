%% ===== PARAMETERS =====
clear; clc; close all;
time_sampling = 1.0;     % [s] receding step
T = 60.0;                % [s] horizon length
N = 20;                  % Chebyshev order
Np = N;
situation = 'cheb_collision_free';
axis_lim = [-200 5000 -200 2800];

% Batasan
r_max = 0.0932;  r_min = -r_max;
u_max = 35*pi/180; u_min = -u_max;  % Konversi manual deg2rad
rrot_max = 7*pi/180; rrot_min = -rrot_max;

% Bobot
weights.w_pos = 5;       weights.w_yaw = 0.02;   weights.w_u = 1e-4;
weights.w_ur = 2e-3;     weights.w_r = 1e-2;
weights.w_pos_T = 20*weights.w_pos;  weights.w_yaw_T = 5*weights.w_yaw;
weights.w_r_T = 5e-2;   weights.w_u_T = 1e-2;  weights.w_tan_T = 5;
weights.R_switch = 250; weights.sigma_sw = 60;

% Parameter kapal
ship_params.Lpp = 101.07; ship_params.B = 14; ship_params.Td = 3.7;
ship_params.m = 2423e3; ship_params.os_surge = 15.4;
ship_params.CB = 0.65; ship_params.xG = 5.25; ship_params.rho = 1024;
ship_params.Adelta = 5.7224; ship_params.gyr = 0.156*ship_params.Lpp;

% Koefisien hidrodinamika
Yvdot = -((1+0.16*ship_params.CB*(ship_params.B/ship_params.Td)-5.1*(ship_params.B/ship_params.Lpp)^2)*pi*(ship_params.Td/ship_params.Lpp)^2);
Yrdot = -((0.67*(ship_params.B/ship_params.Lpp)-0.0033*(ship_params.B/ship_params.Td)^2)*pi*(ship_params.Td/ship_params.Lpp)^2);
Nvdot = -((1.1*(ship_params.B/ship_params.Lpp)-0.041*(ship_params.B/ship_params.Td))*pi*(ship_params.Td/ship_params.Lpp)^2);
Nrdot = -(((1/12)+0.017*(ship_params.CB*ship_params.B/ship_params.Td)-0.33*(ship_params.B/ship_params.Lpp))*pi*(ship_params.Td/ship_params.Lpp)^2);
Yv = -((1+0.4*(ship_params.CB*ship_params.B/ship_params.Td))*pi*(ship_params.Td/ship_params.Lpp)^2);
Yr = -((-0.5+2.2*(ship_params.B/ship_params.Lpp)-0.08*(ship_params.B/ship_params.Td))*pi*(ship_params.Td/ship_params.Lpp)^2);
Nv = -((0.5+2.4*(ship_params.Td/ship_params.Lpp))*pi*(ship_params.Td/ship_params.Lpp)^2);
Nr = -((0.25+0.039*(ship_params.B/ship_params.Td)-0.56*(ship_params.B/ship_params.Lpp))*pi*(ship_params.Td/ship_params.Lpp)^2);

Ydelta = ship_params.rho*pi*ship_params.Adelta/(4*ship_params.Lpp*ship_params.Td);
Ir = (ship_params.m*(ship_params.gyr)^2)/(0.5*ship_params.rho*ship_params.Lpp^5);
Iz = (ship_params.m*(ship_params.xG^2))/(0.5*ship_params.rho*ship_params.Lpp^5) + Ir;

nd_u = 1; nd_m = ship_params.m/(0.5*ship_params.rho*ship_params.Lpp^3); nd_xG = ship_params.xG/ship_params.Lpp;
M = [nd_m-Yvdot, nd_m*nd_xG-Yrdot; nd_m*nd_xG-Nvdot, Iz-Nrdot];
Nmat = [-Yv, nd_m*nd_u-Yr; -Nv, nd_m*nd_xG*nd_u-Nr];
A_lin = -inv(M)*Nmat;

ship_params.a11 = A_lin(1,1); ship_params.a12 = A_lin(1,2);
ship_params.a21 = A_lin(2,1); ship_params.a22 = A_lin(2,2);
b = [0.01; 1];
ship_params.b11 = b(1); ship_params.b12 = b(2)/ship_params.Lpp;

%% ===== CHEBYSHEV SETUP =====
[taus, w_cc, D] = cheb_nodes_weights_D(N);
alpha = 2/T;
n_states = 5; n_controls = 1; n_nodes = N+1;
n_z = n_states*n_nodes + n_controls*n_nodes;

%% ===== BOUNDS =====
lb = -inf(n_z, 1); ub = inf(n_z, 1);
% Yaw rate bounds
for k = 1:n_nodes
    idx_r = (k-1)*n_states + 2;
    lb(idx_r) = r_min; ub(idx_r) = r_max;
end
% Rudder angle bounds
for k = 1:n_nodes
    idx_u = n_states*n_nodes + k;
    lb(idx_u) = u_min; ub(idx_u) = u_max;
end

%% ===== LINEAR INEQUALITY: CONTROL RATE =====
A_rate = zeros(2*n_nodes, n_z); b_rate = zeros(2*n_nodes, 1);
for k = 1:n_nodes
    % ur <= rrot_max
    for i = 1:n_nodes
        idx_u_i = n_states*n_nodes + i;
        A_rate(k, idx_u_i) = alpha * D(k,i);
    end
    b_rate(k) = rrot_max;
    
    % -ur <= -rrot_min  => ur >= rrot_min
    for i = 1:n_nodes
        idx_u_i = n_states*n_nodes + i;
        A_rate(n_nodes+k, idx_u_i) = -alpha * D(k,i);
    end
    b_rate(n_nodes+k) = -rrot_min;
end

%% ===== INITIAL GUESS & SIMULATION =====
x0_val = [0; 0; 0; 0; pi/2];
reference_pose = [4000; 2000; 0];
z0 = zeros(n_z, 1);
array_state = x0_val;
array_state_history = [];
control_sequence = [];
simulation_time = 300;
mpciter = 0;

% Fungsi tujuan dan kendala
cost_func = @(z) mpc_cost_cheb(z, [x0_val; reference_pose], N, T, ship_params, weights);
constraint_func = @(z) mpc_constraints_cheb(z, [x0_val; reference_pose], N, T, ship_params);

% Options fmincon
options = optimoptions('fmincon', 'Display', 'off', 'MaxIterations', 600, ...
                       'MaxFunctionEvaluations', 5000, 'TolCon', 1e-6);

%% ===== MPC LOOP =====
while mpciter < simulation_time/time_sampling
    fprintf('Iterasi %d ...\n', mpciter);
    
    % Update parameter
    p_val = [x0_val; reference_pose];
    cost_func = @(z) mpc_cost_cheb(z, p_val, N, T, ship_params, weights);
    constraint_func = @(z) mpc_constraints_cheb(z, p_val, N, T, ship_params);
    
    % SOLVE NLP
    [z_opt, fval] = fmincon(cost_func, z0, A_rate, b_rate, [], [], lb, ub, ...
                           constraint_func, options);
    
    % Extract solusi
    S_opt = reshape(z_opt(1:n_states*n_nodes), n_states, n_nodes);
    U_opt = reshape(z_opt(n_states*n_nodes+1:end), n_controls, n_nodes);
    
    % Simpan history
    pred_states = [S_opt.'; S_opt(:,end).'];
    array_state_history(:,:,end+1) = pred_states;
    
    % Kontrol diterapkan
    u_apply = U_opt(:,1);
    control_sequence = [control_sequence; u_apply];
    
    % Propagasi
    xdot_now = ship_dynamics(x0_val, u_apply, ship_params);
    next_state = x0_val + time_sampling * xdot_now;
    next_state(5) = mod(next_state(5), 2*pi);
    x0_val = next_state;
    array_state(:,end+1) = x0_val;
    
    % Warm start
    z0 = z_opt;
    
    % Update iterasi
    mpciter = mpciter + 1;
    
    % Cek kondisi berhenti
    dist_nodes = sqrt((S_opt(3,:) - reference_pose(1)).^2 + ...
                       (S_opt(4,:) - reference_pose(2)).^2);
    [dist_min, idx_min] = min(dist_nodes);
    r_at_min = S_opt(2, idx_min);
    u_at_min = U_opt(1, idx_min);
    dist_actual = norm(x0_val(3:4) - reference_pose(1:2));
    
    if (dist_min <= 60 && abs(r_at_min) <= 0.01 && abs(u_at_min) <= 1*pi/180) || ...
       dist_actual <= 60
        fprintf('MISSION COMPLETE pada iterasi %d (dist_min=%.2f m)\n', mpciter, dist_min);
        break;
    end
end

%% ===== OUTPUT =====
T_sim = size(array_state,2);
time = 0:time_sampling:(T_sim-1)*time_sampling;
total_time_index = T_sim + N + 1;
time_plot = linspace(0, time_sampling*(mpciter+Np), total_time_index);

% Padding history terakhir
if isempty(array_state_history)
    last_hist = repmat(array_state(:,end).', Np+2, 1);
else
    last_hist = array_state_history(:,:,end);
end
state_last = last_hist(2:end,:).';
array_state = cat(2, array_state, state_last);

% Visualisasi
draw_collision_free(array_state, array_state_history, ...
    reference_pose, total_time_index, axis_lim, Np, situation, time_plot);