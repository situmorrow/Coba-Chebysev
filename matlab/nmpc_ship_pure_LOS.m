function nmpc_ship_pure_LOS()
% ================================================================
% NMPC kapal (single-shooting) PURE MATLAB — TANPA CasADi/Ipopt
% - Heading pakai Line-of-Sight (LOS) => anti "muter-muter"
% - Batas rudder u dipenuhi via tanh (hard bound)
% - Δu & yaw-rate r dibatasi secara lunak (soft penalty)
% - Hitung waktu komputasi total & per-step (tic/toc)
% ================================================================

clc; clear; close all;

%% ===================== Parameter (basis dari kode Anda) ==================
dt = 1;                 % time sampling
Np = 60;                % prediction horizon

% Batas input & rate
tactical_diameter = 948; %#ok<NASGU>
r_max = 0.0932; r_min = -r_max;          % yaw-rate bounds
u_max = deg2rad(35); u_min = -u_max;     % rudder bounds
rrot_max = deg2rad(5); rrot_min = -rrot_max; % Δu bounds (per step)

%% ======= Data SIGMA (ringkas seperti kode Anda) ==========================
Lpp = 101.07; B = 14; T = 3.7; m = 2423e3; os_surge = 15.4; CB = 0.65;
xG = 5.25; rho = 1024; Adelta = 5.7224; gyration = 0.156*Lpp;

Yvdot = -((1+0.16*CB*(B/T)-5.1*(B/Lpp)^2)*pi*(T/Lpp)^2);
Yrdot = -((0.67*(B/Lpp)-0.0033*(B/T)^2)*pi*(T/Lpp)^2);
Nvdot = -((1.1*(B/Lpp)-0.041*(B/T))*pi*(T/Lpp)^2);
Nrdot = -(((1/12)+0.017*(CB*B/T)-0.33*(B/Lpp))*pi*(T/Lpp)^2);
Yv    = -((1+0.4*(CB*B/T))*pi*(T/Lpp)^2);
Yr    = -((-0.5+2.2*(B/Lpp)-0.08*(B/T))*pi*(T/Lpp)^2);
Nv    = -((0.5+2.4*(T/Lpp))*pi*(T/Lpp)^2);
Nr    = -((0.25+0.039*(B/T)-0.56*(B/Lpp))*pi*(T/Lpp)^2);

Ydelta = rho*pi*Adelta/(4*Lpp*T); %#ok<NASGU>
Ndelta = -0.5*Ydelta;             %#ok<NASGU>
Ir = (m*(gyration^2))/(0.5*rho*Lpp^5);
Iz = (m*(xG^2))/(0.5*rho*Lpp^5)+Ir;

nd_u = 1;
nd_m  = m/(0.5*rho*Lpp^3);
nd_xG = xG/Lpp;

M = [nd_m-Yvdot,           nd_m*nd_xG-Yrdot; ...
     nd_m*nd_xG-Nvdot,     Iz-Nrdot];
N = [-Yv,                  nd_m*nd_u-Yr; ...
     -Nv,                  nd_m*nd_xG*nd_u-Nr];

model_A = -(M\N);
a11 = model_A(1,1); a12 = model_A(1,2);
a21 = model_A(2,1); a22 = model_A(2,2);

b = [0.01; 1];                   % dari penelitian yang Anda pakai
model_B = [b(1); b(2)/Lpp];

% Diskretisasi (Euler pada kontinu di fungsi model), skala seperti kode Anda
os_U = os_surge;
os_A = os_U*[a11 a12*Lpp; a21/Lpp a22]/Lpp;
os_B = os_U^2*model_B/Lpp;

%% ===================== Bobot biaya ==============================
w_position    = 1e-4;   % untuk (x,y)
w_control     = 1e-2;   % untuk u^2
w_orientation = 0.1;    % untuk error heading (LOS)
% penalti keras-lunak:
w_r_bound   = 1e4;      % yaw-rate bound
w_dur_bound = 1e4;      % Δu bound

%% ===================== Inisialisasi MPC =========================
x   = [0; 0; 0; 0; pi/2];          % [v; r; x; y; psi]
ref = [500; 1000; 0];              % [x_ref; y_ref; psi_ref] (psi_ref tdk dipakai utk LOS)

u_prev = 0;
z_opt  = zeros(Np+1,1);            % keputusan di ruang tak-terbatas

max_iter = 300;
dist_tol = 10;                      % toleransi jarak ke target

X_hist = x;
U_hist = [];

% Opsi Nelder–Mead
opts = optimset('Display','off','MaxFunEvals',5e4,'MaxIter',5e4,'TolX',1e-6,'TolFun',1e-6);

%% ===================== Timer (waktu komputasi) ==================
Kmax = max_iter;
time_opt  = zeros(Kmax,1);
time_step = zeros(Kmax,1);
nfev      = zeros(Kmax,1);
global EVAL_COUNT
t_total = tic;
k_idx = 0;

fprintf('RUN NMPC (PURE, LOS)...\n');

%% ===================== Loop NMPC ================================
while norm([x(3); x(4)] - ref(1:2)) > dist_tol && k_idx < max_iter
    k_idx = k_idx + 1;
    t_step = tic;

    costfun = @(z) nmpc_cost_LOS( ...
        z, x, u_prev, ref, os_A, os_B, os_surge, dt, Np, ...
        w_position, w_control, w_orientation, ...
        w_r_bound, w_dur_bound, ...
        r_min, r_max, rrot_min, rrot_max, ...
        u_min, u_max);

    % Warm start
    if k_idx > 1, z_opt = [z_opt(2:end); z_opt(end)]; end

    % Optimisasi
    EVAL_COUNT = 0;
    t_opt = tic;
    [z_opt, ~] = fminsearch(costfun, z_opt, opts);
    time_opt(k_idx) = toc(t_opt);
    nfev(k_idx)     = EVAL_COUNT;

    % Terapkan kontrol pertama
    u_seq = z2u(z_opt, u_min, u_max);
    u = u_seq(1);
    U_hist(end+1,1) = u;

    % Propagasi 1 langkah (RK4)
    x = rk4(@(xx,uu) ship_model_cont(xx,uu,os_A,os_B,os_surge), x, u, dt);
    x(5) = wrapToPi_local(x(5));
    X_hist(:,end+1) = x;

    % Update
    u_prev = u;
    time_step(k_idx) = toc(t_step);
end

T_total = toc(t_total);

%% ===================== Laporan waktu ============================
valid = time_opt(1:k_idx);
fprintf('\n===== WAKTU KOMPUTASI =====\n');
fprintf('Total NMPC              : %.4f s\n', T_total);
fprintf('Rata2 optimisasi        : %.4f s/step\n', mean(valid));
fprintf('Median optimisasi       : %.4f s/step\n', median(valid));
fprintf('Maks optimisasi         : %.4f s (step %d)\n', max(valid), find(valid==max(valid),1));
fprintf('Rata2 total per-step    : %.4f s/step\n', mean(time_step(1:k_idx)));
fprintf('Eval cost (mean per step): %.1f panggilan\n', mean(nfev(1:k_idx)));

%% ===================== Plot ================================
t = 0:dt:(size(X_hist,2)-1)*dt;

figure('Color','w');
subplot(3,1,1); hold on; grid on; axis equal;
plot(X_hist(3,:), X_hist(4,:), 'LineWidth',1.6);
plot(0,0,'go','MarkerFaceColor','g'); plot(ref(1),ref(2),'r*','MarkerSize',10);
xlabel('x [m]'); ylabel('y [m]'); title('Lintasan');

subplot(3,1,2); hold on; grid on;
plot(t, X_hist(5,:), 'LineWidth',1.6);
ylabel('\psi [rad]'); title('Heading');

subplot(3,1,3); hold on; grid on;
stairs(t(1:end-1), U_hist, 'LineWidth',1.6);
yline(u_min,'--r'); yline(u_max,'--r');
ylabel('rudder [rad]'); xlabel('time [s]'); title('Input rudder');

% (opsional) animasi trail singkat bisa ditambah seperti contoh sebelumnya

end
%% ===================== COST (dengan LOS) =========================
function J = nmpc_cost_LOS(z, x0, u_prev, ref, ...
                           os_A, os_B, os_surge, dt, Np, ...
                           w_position, w_control, w_orientation, ...
                           w_r_bound, w_dur_bound, ...
                           r_min, r_max, rrot_min, rrot_max, ...
                           u_min, u_max)
global EVAL_COUNT
EVAL_COUNT = EVAL_COUNT + 1;

u_seq = z2u(z, u_min, u_max);
x = x0;  J = 0;

for k = 1:Np+1
    u = u_seq(k);
    % Prediksi 1 langkah (Euler di fungsi, tapi kita pakai RK4 untuk akurasi)
    x_next = rk4(@(xx,uu) ship_model_cont(xx,uu,os_A,os_B,os_surge), x, u, dt);
    x_next(5) = wrapToPi_local(x_next(5));

    if k <= Np
        % ---- LOS: heading desired menghadap target posisi ----
        e_xy   = [ref(1)-x_next(3); ref(2)-x_next(4)];
        psi_des= atan2(e_xy(2), e_xy(1));
        e_psi  = wrapToPi_local(psi_des - x_next(5));

        % Stage cost
        J = J ...
          + w_position * (e_xy.'*e_xy) ...
          + w_orientation * (e_psi^2) ...
          + w_control * (u^2);

        % Δu soft-bound
        if k==1, du = u - u_prev; else, du = u - u_seq(k-1); end
        if du < rrot_min, J = J + w_dur_bound*(rrot_min - du)^2; end
        if du > rrot_max, J = J + w_dur_bound*(du - rrot_max)^2; end
    end

    % yaw-rate r soft-bound
    r = x_next(2);
    if r < r_min, J = J + w_r_bound*(r_min - r)^2; end
    if r > r_max, J = J + w_r_bound*(r - r_max)^2; end

    x = x_next;
end

% Terminal LOS
eT_xy   = [ref(1)-x(3); ref(2)-x(4)];
psi_des_T = atan2(eT_xy(2), eT_xy(1));
eT_psi  = wrapToPi_local(psi_des_T - x(5));
J = J + 10*w_position*(eT_xy.'*eT_xy) + 10*w_orientation*(eT_psi^2);
end
%% ===================== MODEL KAPAL (kontinu) =====================
function xdot = ship_model_cont(x, u, os_A, os_B, os_surge)
% x = [v; r; x; y; psi]
v  = x(1); r = x(2); psi = x(5); %#ok<NASGU>

os_xdot_dyn = os_A*[v; r] + os_B*u;  % [vdot; rdot]
vdot = os_xdot_dyn(1);
rdot = os_xdot_dyn(2);

xdot = [ vdot; ...
         rdot; ...
         os_surge*cos(x(5)) - v*sin(x(5)); ...
         os_surge*sin(x(5)) + v*cos(x(5)); ...
         r ];
end
%% ===================== RK4 integrator ============================
function x_next = rk4(f, x, u, h)
    k1 = f(x, u);
    k2 = f(x + 0.5*h*k1, u);
    k3 = f(x + 0.5*h*k2, u);
    k4 = f(x + h*k3, u);
    x_next = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end
%% ===================== z -> u (hard bound) =======================
function u = z2u(z, u_min, u_max)
u = u_min + (u_max - u_min)*(tanh(z) + 1)/2;
end
%% ===================== wrap sudut ================================
function ang = wrapToPi_local(ang)
ang = mod(ang + pi, 2*pi) - pi;
end
