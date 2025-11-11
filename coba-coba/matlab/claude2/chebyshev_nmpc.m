%% Chebyshev Pseudospectral NMPC - Ship Point Stabilization (Fix: LOS + atan2)
% Pure MATLAB implementation (tanpa CasADi)

clc; clear; close all;

%% ================== PARAMETER UTAMA ==================
time_sampling   = 1.0;       % detik (MPC update rate)
N               = 20;        % titik kolokasi CGL (15–25 disarankan)

% Batas fisik
tactical_diameter = 948;
r_max = 0.0932; r_min = -r_max;           % batas yaw-rate (rad/s)
u_max = deg2rad(35); u_min = -u_max;      % batas sudut kemudi (rad)
rrot_max = deg2rad(5); rrot_min = -rrot_max; % batas laju kemudi (rad/s)

% Bobot biaya
w_position    = 5e-2;      % posisi
w_control     = 1e-3;      % besaran kontrol
w_orientation = 4e-1;      % orientasi (heading)

% Look-ahead LOS (meter)
L_LOS = 50;  % bisa dibuat fungsi jarak atau U*Ts, diset konstan dulu

% Simulasi
simulation_time      = 300;            % detik (maks iterasi)
distance_condition   = 10;             % m (jarak selesai)

%% ================== MODEL KAPAL (SIGMA) ==================
Lpp = 101.07; B = 14; T = 3.7; m = 2423*1e3;
os_surge = 15.4; CB = 0.65; xG = 5.25; rho = 1024;
Adelta = 5.7224; gyration = 0.156*Lpp;

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

nd_u = 1; nd_m = m/(0.5*rho*Lpp^3); nd_xG = xG/Lpp;

M = [nd_m-Yvdot nd_m*nd_xG-Yrdot;
     nd_m*nd_xG-Nvdot Iz-Nrdot];
Nn = [-Yv nd_m*nd_u-Yr;
      -Nv nd_m*nd_xG*nd_u-Nr];

model_A = -M\Nn;
a11 = model_A(1,1); a12 = model_A(1,2);
a21 = model_A(2,1); a22 = model_A(2,2);

b = [0.01; 1]; b11 = b(1); b12 = b(2);
model_B = [b11; b12/Lpp];

os_U = os_surge;
os_A = os_U*[a11 a12*Lpp; a21/Lpp a22]/Lpp;
os_B = os_U^2*model_B/Lpp;

%% ================== CHEBYSHEV SETUP ==================
% Titik CGL
tau = flipud(cos((0:N)'*pi/N)); % dari -1 ke 1
% Matriks diferensiasi & bobot kuadratur
D = chebyshev_differentiation_matrix(N);
w = chebyshev_weights(N);

%% ================== INISIALISASI MPC ==================
x0 = [0; 0; 0; 0; pi/2];             % [v, r, x, y, psi]
reference_pose = [500; 2000; 0];     % [x_ref, y_ref, psi_ref] (psi_ref tak dipakai)

array_state = x0;
control_sequence = [];
mpciter = 0; z_previous = [];

distance_to_destination = norm(x0(3:4) - reference_pose(1:2));

main_tic = tic;
while (distance_to_destination > distance_condition && ...
       mpciter < simulation_time/time_sampling)

    % ---- Horizon adaptif supaya target "terlihat" ----
    distance_now = norm(x0(3:4) - reference_pose(1:2));
    time_horizon = min(150, max(25, 1.5*distance_now/os_surge));

    fprintf('MPC Iteration: %d, Dist: %.1f m, Horizon: %.1f s\n', ...
        mpciter, distance_now, time_horizon);

    % ---- Selesaikan NMPC pseudospektral ----
    [u_opt, X_opt, z_previous] = solve_chebyshev_nmpc( ...
        x0, reference_pose, N, tau, D, w, time_horizon, ...
        os_A, os_B, os_surge, r_min, r_max, u_min, u_max, ...
        rrot_min, rrot_max, w_position, w_control, w_orientation, ...
        L_LOS, z_previous);

    % ---- Terapkan kontrol pertama (clamp laju kemudi nyata) ----
    u_current = u_opt(1);
    if ~isempty(control_sequence)
        du = u_current - control_sequence(end);
        du = max(min(du, rrot_max*time_sampling), rrot_min*time_sampling);
        u_current = control_sequence(end) + du;
    end
    control_sequence = [control_sequence; u_current];

    % ---- Simulasi 1 step (Euler) ----
    xdot = ship_dynamics(x0, u_current, os_A, os_B, os_surge);
    x_next = x0 + time_sampling * xdot;
    x_next(5) = mod(x_next(5), 2*pi); % wrap heading

    x0 = x_next;
    array_state = [array_state, x0];

    distance_to_destination = norm(x0(3:4) - reference_pose(1:2));
    mpciter = mpciter + 1;
end

avg_time = toc(main_tic) / max(1, mpciter);
fprintf('\n=== Simulation Complete ===\nFinal distance: %.2f m\nAvg MPC time: %.4f s\nIters: %d\n', ...
    distance_to_destination, avg_time, mpciter);

%% ================== PLOT ==================
time = 0:time_sampling:(mpciter*time_sampling);
plot_results(time, array_state, control_sequence, reference_pose, ...
    r_min, r_max, u_min, u_max, rrot_min, rrot_max);

%% ================== ===== FUNCTIONS ===== ==================

function D = chebyshev_differentiation_matrix(N)
    D = zeros(N+1);
    tau = flipud(cos((0:N)'*pi/N));
    c = ones(N+1,1); c([1 end]) = 2;
    for i = 0:N
        for j = 0:N
            if i == j
                if i == 0
                    D(i+1,j+1) = (2*N^2 + 1)/6;
                elseif i == N
                    D(i+1,j+1) = -(2*N^2 + 1)/6;
                else
                    D(i+1,j+1) = -tau(i+1)/(2*(1 - tau(i+1)^2));
                end
            else
                D(i+1,j+1) = (c(i+1)/c(j+1)) * (-1)^(i+j) / (tau(i+1)-tau(j+1));
            end
        end
    end
end

function w = chebyshev_weights(N)
    % Bobot kuadratur CGL (Clenshaw–Curtis)
    w = zeros(N+1,1);
    if mod(N,2)==0
        w([1,end]) = 1/(N^2-1);
        for k=1:N-1
            s = gamma_func(0,k,N) + gamma_func(N/2,k,N);
            for i=1:(N/2-1), s = s + 2*gamma_func(i,k,N); end
            w(k+1) = (2/N)*s;
        end
    else
        w([1,end]) = 1/N^2;
        for k=1:N-1
            s = gamma_func(0,k,N);
            for i=1:((N-1)/2), s = s + 2*gamma_func(i,k,N); end
            w(k+1) = (2/N)*s;
        end
    end
end

function g = gamma_func(i,k,N)
    g = (1/(1-4*i^2))*cos(2*pi*i*k/N);
end

function xdot = ship_dynamics(x, u, os_A, os_B, os_surge)
    % x = [v, r, x_ship, y_ship, psi]
    v = x(1); r = x(2); psi = x(5);
    vr = os_A * [v; r] + os_B * u;
    xdot = [ vr(1);
             vr(2);
             os_surge*cos(psi) - v*sin(psi);
             os_surge*sin(psi) + v*cos(psi);
             r ];
end

function psi_ref = los_heading(xs, ys, xg, yg, L)
    % Look-ahead LOS: virtual point L meter ke arah goal
    dx = xg - xs; dy = yg - ys;
    dist = hypot(dx,dy) + 1e-9;
    ex = dx/dist; ey = dy/dist;
    xla = xs + L*ex; yla = ys + L*ey;
    psi_ref = atan2(yla - ys, xla - xs);
end

function diff = angle_difference(a1, a2)
    % beda sudut terkecil (−pi..pi) berbasis atan2
    diff = atan2(sin(a1 - a2), cos(a1 - a2));
end

function [u_opt, X_opt, z_opt] = solve_chebyshev_nmpc( ...
    x0, ref, N, tau, D, w, T, os_A, os_B, os_surge, ...
    r_min, r_max, u_min, u_max, rrot_min, rrot_max, ...
    w_pos, w_ctrl, w_orient, L_LOS, z_prev)

    n_states = 5; n_controls = 1;
    n_vars = n_states*(N+1) + n_controls*(N+1);

    % ---------- Initial guess ----------
    if isempty(z_prev)
        z0 = zeros(n_vars,1);
        for i = 0:N
            alpha = i/N;
            xi = x0(3) + alpha*(ref(1)-x0(3));
            yi = x0(4) + alpha*(ref(2)-x0(4));
            psi_i = los_heading(xi, yi, ref(1), ref(2), L_LOS);
            z0(i*n_states + (1:n_states)) = [0;0;xi;yi;psi_i];
        end
        z0(n_states*(N+1)+1:end) = 0; % kontrol awal
    else
        z0 = z_prev;
        z0(1:n_states) = x0;
    end

    % ---------- Bounds ----------
    lb = -inf(n_vars,1); ub = inf(n_vars,1);
    % kontrol
    for i=0:N
        idx = n_states*(N+1) + i + 1;
        lb(idx) = u_min; ub(idx) = u_max;
    end
    % yaw-rate
    for i=0:N
        idx_r = i*n_states + 2;
        lb(idx_r) = r_min; ub(idx_r) = r_max;
    end

    % ---------- Objective & Constraints ----------
    obj_fun  = @(z) objective_function(z,N,w,T,ref,w_pos,w_ctrl,w_orient,L_LOS);
    nonlcon  = @(z) nonlinear_constraints_full(z,N,D,T,x0,ref,os_A,os_B,os_surge, ...
                                               rrot_min,rrot_max,L_LOS);

    opts = optimoptions('fmincon','Display','off','Algorithm','sqp', ...
        'MaxIterations',2000,'MaxFunctionEvaluations',20000, ...
        'ConstraintTolerance',1e-4,'OptimalityTolerance',1e-4, ...
        'StepTolerance',1e-8,'FiniteDifferenceStepSize',1e-6);

    [z_opt,fval,exitflag,output] = fmincon(obj_fun,z0,[],[],[],[],lb,ub,nonlcon,opts);
    if exitflag<=0
        warning('Optimization did not converge. Exit flag: %d', exitflag);
    end

    [c_chk,ceq_chk] = nonlcon(z_opt);
    fprintf('Cost: %.3e | Ineq viol: %.1e | Eq viol: %.1e | Iter: %d\n', ...
        fval, max([0;c_chk]), max(abs(ceq_chk)), output.iterations);

    X_opt = reshape(z_opt(1:n_states*(N+1)), n_states, N+1);
    u_opt = z_opt(n_states*(N+1)+1:end);
end

function J = objective_function(z,N,w,T,ref,w_pos,w_ctrl,w_orient,L_LOS)
    n_states = 5;
    X = reshape(z(1:n_states*(N+1)), n_states, N+1);
    U = z(n_states*(N+1)+1:end);

    % bobot tambahan
    w_rate = 5e-3; w_sway = 1e-2; w_yaw = 5e-3;

    J = 0;
    for i=0:N
        v = X(1,i+1); r = X(2,i+1);
        xs = X(3,i+1); ys = X(4,i+1); psi = X(5,i+1);

        psi_ref = los_heading(xs,ys,ref(1),ref(2),L_LOS);
        heading_error = angle_difference(psi, psi_ref);

        pos_error = [xs;ys] - ref(1:2);
        u = U(i+1);

        stage = w_pos*(pos_error.'*pos_error) + ...
                w_orient*heading_error^2 + ...
                w_ctrl*u^2 + w_sway*v^2 + w_yaw*r^2;

        J = J + w(i+1)*stage;
    end

    for i=1:N
        du = U(i+1)-U(i);
        J = J + w_rate*w(i+1)*du^2;
    end

    terminal_factor = 80;
    xf = X(3,end); yf = X(4,end); psif = X(5,end);
    posf = [xf;yf] - ref(1:2);
    psi_ref_f = los_heading(xf,yf,ref(1),ref(2),L_LOS);
    hef = angle_difference(psif, psi_ref_f);
    J = J + terminal_factor*( w_pos*(posf.'*posf) + w_orient*hef^2 );

    J = (T/2)*J;
end

function [c,ceq] = nonlinear_constraints_full(z,N,D,T,x0,ref,os_A,os_B,os_surge, ...
                                              rrot_min,rrot_max,L_LOS)
    n_states = 5;
    X = reshape(z(1:n_states*(N+1)), n_states, N+1);
    U = z(n_states*(N+1)+1:end);

    ceq = [];
    % kondisi awal
    ceq = [ceq; X(:,1) - x0];

    % kolokasi (k=1..N)
    for k=1:N
        xdot = zeros(n_states,1);
        for j=0:N
            xdot = xdot + D(k+1,j+1)*X(:,j+1);
        end
        xdot = (2/T)*xdot;

        xk = X(:,k+1); uk = U(k+1);
        fk = ship_dynamics(xk, uk, os_A, os_B, os_surge);

        ceq = [ceq; xdot - fk];
    end

    % --------- INEQUALITY ---------
    c = [];

    % batas laju kemudi (via Chebyshev differentiation)
    for k=1:N
        udot = 0;
        for j=0:N
            udot = udot + D(k+1,j+1)*U(j+1);
        end
        udot = (2/T)*udot;
        c = [c; udot - rrot_max];            % udot <= rrot_max
        c = [c; -udot - (-rrot_min)];        % udot >= rrot_min
    end

    % No-backtracking: kecepatan proyeksi ke arah goal >= 0
    for k=1:N
        xk = X(:,k+1); uk = U(k+1);
        fk = ship_dynamics(xk, uk, os_A, os_B, os_surge);
        dxg = ref(1) - xk(3); dyg = ref(2) - xk(4);
        distg = hypot(dxg, dyg) + 1e-9;
        ex = dxg/distg; ey = dyg/distg;
        v_along = fk(3)*ex + fk(4)*ey;  % proyeksi kecepatan posisi
        c = [c; -v_along];              % v_along >= 0  --> -v_along <= 0
    end
end

function plot_results(time, states, controls, ref, ...
    r_min, r_max, u_min, u_max, rrot_min, rrot_max)

    lw = 1.5; fs = 12;

    figure('Position',[100 100 800 600]);
    plot(states(3,:), states(4,:), 'b-','LineWidth',2); hold on;
    plot(states(3,1), states(4,1), 'go','MarkerSize',10,'MarkerFaceColor','g');
    plot(ref(1), ref(2), 'rx','MarkerSize',15,'LineWidth',3);
    grid on; xlabel('X Position (m)'); ylabel('Y Position (m)');
    title('Ship Trajectory','FontSize',fs+2); legend('Trajectory','Start','Target');
    axis equal;

    figure('Position',[100 100 1200 800]);

    subplot(3,2,1);
    plot(time, states(1,:), 'b-','LineWidth',lw); grid on;
    xlabel('Time (s)'); ylabel('Sway v (m/s)'); title('Sway Velocity');

    subplot(3,2,2);
    plot(time, states(2,:), 'b-','LineWidth',lw); hold on;
    plot(time, r_max*ones(size(time)), 'r--','LineWidth',lw);
    plot(time, r_min*ones(size(time)), 'r--','LineWidth',lw);
    grid on; xlabel('Time (s)'); ylabel('\omega_z (rad/s)');
    title('Yaw Rate with Constraints'); legend('r','limits','Location','best');

    subplot(3,2,3);
    plot(time, states(3,:), 'b-','LineWidth',lw); hold on;
    plot(time, ref(1)*ones(size(time)), 'r--','LineWidth',lw);
    grid on; xlabel('Time (s)'); ylabel('X (m)'); title('X Position');
    legend('Actual','Ref');

    subplot(3,2,4);
    plot(time, states(4,:), 'b-','LineWidth',lw); hold on;
    plot(time, ref(2)*ones(size(time)), 'r--','LineWidth',lw);
    grid on; xlabel('Time (s)'); ylabel('Y (m)'); title('Y Position');
    legend('Actual','Ref');

    subplot(3,2,5);
    plot(time, rad2deg(states(5,:)), 'b-','LineWidth',lw); hold on;
    grid on; xlabel('Time (s)'); ylabel('\psi (deg)'); title('Heading');

    subplot(3,2,6);
    ct = time(1:end-1);
    stairs(ct, rad2deg(controls), 'k-','LineWidth',lw); hold on;
    plot(ct, rad2deg(u_max)*ones(size(ct)), 'r--','LineWidth',lw);
    plot(ct, rad2deg(u_min)*ones(size(ct)), 'r--','LineWidth',lw);
    grid on; xlabel('Time (s)'); ylabel('Rudder (deg)');
    title('Control Input'); legend('u','limits','Location','best');

    if numel(controls) > 1
        figure('Position',[100 100 600 400]);
        cr = diff(rad2deg(controls));
        stairs(ct(2:end), cr, 'k-','LineWidth',lw); hold on;
        plot(ct(2:end), rad2deg(rrot_max)*ones(numel(ct)-1,1), 'r--','LineWidth',lw);
        plot(ct(2:end), rad2deg(rrot_min)*ones(numel(ct)-1,1), 'r--','LineWidth',lw);
        grid on; xlabel('Time (s)'); ylabel('Rudder Rate (deg/s)');
        title('Control Rate of Change'); legend('Rate','Constraints');
    end
end
