function nmpc_ship_pure()
% ================================================================
% NMPC kapal (point stabilization, single-shooting) — PURE MATLAB
% Pengganti CasADi/Ipopt: pakai fminsearch + penalti kendala.
% Referensi: skrip CasADi pengguna (model & bobot).  (lihat komentar)
% ================================================================

clc; clear; close all;

%% ===================== Parameter (dari skrip CasADi) =====================
dt = 1;                 % time_sampling
Np = 60;                % prediction horizon

% Batas input & rate (rudder)
tactical_diameter = 948; %#ok<NASGU>
r_max = 0.0932; r_min = -r_max;     % yaw-rate bounds
u_max = deg2rad(35); u_min = -u_max;
rrot_max = deg2rad(5); rrot_min = -rrot_max;

% ----- State: [v;r;x;y;psi] -----
% Parameter SIGMA-class (diringkas; sama seperti skrip CasADi)
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
nd_m = m/(0.5*rho*Lpp^3); nd_xG = xG/Lpp;

M = [nd_m-Yvdot,          nd_m*nd_xG-Yrdot; ...
     nd_m*nd_xG-Nvdot,    Iz-Nrdot];
Nmat = [-Yv,              nd_m*nd_u-Yr; ...
        -Nv,              nd_m*nd_xG*nd_u-Nr];

A = - (M\Nmat);
a11=A(1,1); a12=A(1,2); a21=A(2,1); a22=A(2,2);
b  = [0.01;1];
model_B = [b(1); b(2)/Lpp];    % sesuai skrip

% Discrete-time approx (Euler pada model kontinu di bawah)
os_U = os_surge;
os_A = os_U*[a11 a12*Lpp; a21/Lpp a22]/Lpp;
os_B = os_U^2*model_B/Lpp;

% ======== Bobot biaya (sama basisnya) ========
w_position    = 1e-4;     % untuk (x,y)
w_control     = 1e-2;     % untuk |u|^2
w_orientation = 0.1;      % untuk psi
% tambahkan penalti kendala (besar agar "mendekati" hard constraint)
w_r_bound   = 1e4;        % yaw-rate bound penalty
w_dur_bound = 1e4;        % rudder-rate bound penalty

%% ===================== Inisialisasi MPC ===========================
x = [0;0;0;0;pi/2];               % [v;r;x;y;psi] initial
ref = [500;1000;0];               % reference [x_ref;y_ref;psi_ref]
axis_lim = [-200 3600 -200 2200]; %#ok<NASGU>

u_prev = 0;
u_seq_guess = zeros(Np+1,1);      % tebakan awal (ruang z—lihat z2u)
z_opt = zeros(Np+1,1);

max_iter = 10000;  % loop kontrol maksimum
dist_tol = 10;
mpc_k = 0;

X_hist = x;
U_hist = [];
pred_hist_last = [];

% Opsi Nelder–Mead
opts = optimset('Display','off','MaxFunEvals',5e4,'MaxIter',5e4,'TolX',1e-6,'TolFun',1e-6);

fprintf('RUN NMPC pure MATLAB (no CasADi)...\n');

while norm([x(3);x(4)] - ref(1:2)) > dist_tol && mpc_k < max_iter

    % ===== Cost function untuk horizon sekarang =====
    costfun = @(z) nmpc_cost(z, x, u_prev, ref, ...
                     os_A,os_B,os_surge, dt, Np, ...
                     w_position,w_control,w_orientation, ...
                     w_r_bound, w_dur_bound, ...
                     r_min,r_max, rrot_min,rrot_max, ...
                     u_min,u_max);

    % ===== Optimasi =====
    % warm start: geser solusi sebelumnya
    if mpc_k>0, z_opt = [z_opt(2:end); z_opt(end)]; end
    [z_opt, ~] = fminsearch(costfun, z_opt, opts);

    % ===== Terapkan kontrol pertama =====
    u_seq = z2u(z_opt, u_min, u_max);
    u = u_seq(1);
    U_hist(end+1,1) = u;

    % ===== Simulasi 1 langkah (Euler) =====
    xdot = ship_model_cont(x, u, os_A, os_B, os_surge);
    x = rk4(@(xx,uu) ship_model_cont(xx,uu,os_A,os_B,os_surge), x, u, dt);
    x(5) = wrapToPi_local(x(5));   % wrap psi
    X_hist(:,end+1) = x;

    % ===== Simpan prediksi terakhir (opsional untuk plot) =====
    pred_hist_last = rollout_pred(x - dt*xdot, z_opt, os_A,os_B,os_surge, dt, Np); %#ok<NASGU>

    % ===== Update =====
    u_prev = u;
    mpc_k  = mpc_k + 1;
end

%% ===================== Plot Hasil ==========================
t = 0:dt:(size(X_hist,2)-1)*dt;

figure('Color','w');
subplot(3,1,1); hold on; grid on; axis equal;
plot(X_hist(3,:), X_hist(4,:), 'LineWidth',1.5);
plot(0,0,'go','MarkerFaceColor','g'); plot(ref(1),ref(2),'r*','MarkerSize',10);
xlabel('x [m]'); ylabel('y [m]'); title('Lintasan');

subplot(3,1,2); hold on; grid on;
plot(t, X_hist(5,:), 'LineWidth',1.5);
yline(ref(3),'--r'); ylabel('\psi [rad]'); title('Heading');

subplot(3,1,3); hold on; grid on;
stairs(t(1:end-1), U_hist, 'LineWidth',1.5);
yline(u_min,'--r'); yline(u_max,'--r');
ylabel('rudder [rad]'); xlabel('time [s]');

end

%% ===================== COST FUNCTION ==============================
function J = nmpc_cost(z, x0, u_prev, ref, ...
                       os_A,os_B,os_surge, dt, Np, ...
                       w_pos,w_u,w_psi, ...
                       w_r_bound, w_dur_bound, ...
                       r_min,r_max, rrot_min,rrot_max, ...
                       u_min,u_max)

u_seq = z2u(z, u_min, u_max);
x = x0;  J = 0;

for k = 1:Np+1
    u = u_seq(k);
    xdot   = ship_model_cont(x, u, os_A, os_B, os_surge);
    x_next = x + dt*xdot;
    x_next(5) = wrapToPi_local(x_next(5));

    if k <= Np
        % ---- LOS tracking ----
        e_xy   = [ref(1)-x_next(3); ref(2)-x_next(4)];
        psi_des= atan2(ref(2) - x_next(4), ref(1) - x_next(3));
        e_psi  = wrapToPi_local(psi_des - x_next(5));

        J = J + w_pos*(e_xy.'*e_xy) + w_psi*(e_psi^2) + w_u*(u^2);

        % ---- Δu bounds (soft) ----
        if k==1, du = u - u_prev; else, du = u - u_seq(k-1); end
        if du < rrot_min, J = J + w_dur_bound*(rrot_min - du)^2; end
        if du > rrot_max, J = J + w_dur_bound*(du - rrot_max)^2; end
    end

    % ---- yaw-rate bounds (soft) ----
    r = x_next(2);
    if r < r_min, J = J + w_r_bound*(r_min - r)^2; end
    if r > r_max, J = J + w_r_bound*(r - r_max)^2; end

    x = x_next;
end

% ---- terminal LOS ----
eT_xy   = [ref(1)-x(3); ref(2)-x(4)];
psi_des_T = atan2(ref(2) - x(4), ref(1) - x(3));
eT_psi  = wrapToPi_local(psi_des_T - x(5));
J = J + 10*w_pos*(eT_xy.'*eT_xy) + 10*w_psi*(eT_psi^2);
end
%% ===================== MODEL KONTINU ==============================
function xdot = ship_model_cont(x, u, os_A, os_B, os_surge)
% x = [v;r;x;y;psi]
v  = x(1); r = x(2); psi = x(5);

os_xdot_dyn = os_A*[v; r] + os_B*u;     % 2x1
vdot = os_xdot_dyn(1);
rdot = os_xdot_dyn(2);
xdot = [ vdot;
         rdot;
         os_surge*cos(psi) - v*sin(psi);
         os_surge*sin(psi) + v*cos(psi);
         r ];
end

%% ===================== UTIL: prediksi penuh (opsional plot) =======
function Xpred = rollout_pred(x0, z, os_A,os_B,os_surge, dt, Np)
u_seq = z2u(z, -inf, inf); %#ok<NASGU> % (tidak dipakai disini)
Xpred = zeros(5, Np+2); Xpred(:,1) = x0;
x = x0;
for k=1:Np+1
    u = z2u(z(k), -inf, inf);  % hanya untuk tracing
    x = x + dt*ship_model_cont(x, u, os_A, os_B, os_surge);
    x(5) = wrapToPi_local(x(5));
    Xpred(:,k+1) = x;
end
end

%% ===================== UTIL: z -> u (hard bound) ==================
function u = z2u(z, u_min, u_max)
if isfinite(u_min) && isfinite(u_max)
    u = u_min + (u_max - u_min) * (tanh(z) + 1)/2;
else
    u = z; % jika dipanggil hanya untuk tracing
end
end

%% ===================== UTIL: wrap sudut ===========================
function ang = wrapToPi_local(ang)
ang = mod(ang + pi, 2*pi) - pi;
end
%% Runge Kutta-4
function x_next = rk4(f, x, u, h)
    k1 = f(x, u);
    k2 = f(x + 0.5*h*k1, u);
    k3 = f(x + 0.5*h*k2, u);
    k4 = f(x + h*k3, u);
    x_next = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end
