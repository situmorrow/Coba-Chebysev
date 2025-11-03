function nmpc_ship_LOS_waypoints()
% =====================================================================
% NMPC kapal sederhana — PURE MATLAB (tanpa CasADi/Ipopt/Opt Toolbox)
% - Tracking pakai LOS (Line-of-Sight) ke waypoint aktif (anti "muter")
% - Multi-waypoint path following (ganti target saat dekat)
% - Input rudder u dibatasi keras via tanh (hard bound tanpa solver berbatas)
% - Δu (rudder-rate) & yaw-rate r dibatasi lunak (soft penalty)
% - Integrator RK4; optimizer fminsearch (Nelder-Mead, bawaan MATLAB)
% =====================================================================

clc; clear; close all;

%% ========== PARAMETER MODEL KAPAL (ringkas, konsisten dengan literatur) ==========
% State: x = [v; r; X; Y; psi]
% v   : sway velocity (non-dim)
% r   : yaw rate (non-dim)
% X,Y : posisi (meter)
% psi : heading (rad)
% Dinamika kecepatan lateral & yaw diambil dari model linearized, posisi dari geometri.

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
N  = [-Yv,                  nd_m*nd_u-Yr; ...
      -Nv,                  nd_m*nd_xG*nd_u-Nr];

A = -(M\N);                       % model lateral-yaw (non-dim)
a11 = A(1,1); a12 = A(1,2);
a21 = A(2,1); a22 = A(2,2);

b = [0.01; 1];                    % gaya rudder (disederhanakan)
Bv = [b(1); b(2)/Lpp];

% Skala kecepatan seret sesuai "os_surge"
os_U = os_surge;
os_A = os_U*[a11 a12*Lpp; a21/Lpp a22]/Lpp;
os_B = os_U^2*Bv/Lpp;

%% ========== MPC SETUP ==========
dt  = 1.0;          % time step [s]
Np  = 30;           % prediction horizon (panjang prediksi)
Tsim = 180;         % lama simulasi (detik) — silakan sesuaikan

% Batas input & rate
u_max = deg2rad(35);  u_min = -u_max;      % batas rudder
du_max = deg2rad(6);  du_min = -du_max;    % batas Δu per step
r_max  = 0.0932;      r_min  = -r_max;     % batas yaw-rate (soft)

% Bobot biaya
Qpos = diag([12, 12]);   % penalti posisi (X,Y)
Qpsi = 6;                % penalti error heading (LOS)
Rdu  = 0.30;             % penalti perubahan kontrol Δu
Ru   = 0.05;             % penalti usaha kontrol u
Qr   = 0.80;             % penalti yaw-rate r
P    = diag([60, 60]);   % terminal posisi
Ppsi = 30;               % terminal heading (LOS)

% Penalti soft-constraint
W_du = 1e4;   % untuk pelanggaran Δu
W_r  = 1e4;   % untuk pelanggaran r

% Waypoints (ubah sesuai kebutuhan; setiap baris = [Xref Yref] meter)
W = [ 500 1000;
     1200 1400;
     1800 1400;
     2200 1700];

wp_idx = 1;          % mulai dari waypoint pertama
wp_rad = 30;         % radius switch waypoint [m]

% State awal
x = [0; 0; 0; 0; pi/2];  % [v; r; X; Y; psi]
u_prev = 0;

% History
X_hist = x;
U_hist = [];

% Optimizer
z_opt = zeros(Np+1,1);  % keputusan di ruang "z" (tak-terbatas); u = squash(z)
opts  = optimset('Display','off','MaxFunEvals',3e4,'MaxIter',3e4,'TolX',1e-6,'TolFun',1e-6);

%% ========== SIMULASI NMPC ==========
n_steps = round(Tsim/dt);
for k = 1:n_steps

    % ---- Check switching waypoint berdasarkan posisi sekarang ----
    goal = W(wp_idx, :).';       % [Xg; Yg]
    if norm([x(3);x(4)] - goal) < wp_rad
        if wp_idx < size(W,1)
            wp_idx = wp_idx + 1;
            goal = W(wp_idx, :).';
        else
            % Sudah di waypoint terakhir; boleh keluar lebih awal jika dekat
            if norm([x(3);x(4)] - goal) < wp_rad/2
                disp('Selesai: waypoint terakhir tercapai.'); 
                break;
            end
        end
    end

    % ---- Cost function (LOS ke waypoint aktif) ----
    costfun = @(z) cost_LOS_waypoint( ...
        z, x, u_prev, goal, ...
        os_A, os_B, os_surge, dt, Np, ...
        Qpos, Qpsi, Rdu, Ru, Qr, P, Ppsi, ...
        du_min, du_max, W_du, ...
        r_min, r_max, W_r, ...
        u_min, u_max);

    % ---- Warm-start & Optimisasi ----
    if k > 1, z_opt = [z_opt(2:end); z_opt(end)]; end
    [z_opt, ~] = fminsearch(costfun, z_opt, opts);

    % ---- Terapkan kontrol pertama ----
    u_seq = z2u(z_opt, u_min, u_max);
    u = u_seq(1);
    U_hist(end+1,1) = u;

    % ---- Propagasi 1 langkah (RK4) ----
    x = rk4(@(xx,uu) ship_dyn(xx,uu,os_A,os_B,os_surge), x, u, dt);
    x(5) = wrapToPi_local(x(5));
    X_hist(:,end+1) = x;
    u_prev = u;
end

%% ========== PLOT HASIL ==========
t = 0:dt:(size(X_hist,2)-1)*dt;

figure('Color','w');
subplot(3,1,1); hold on; grid on; axis equal;
plot(X_hist(3,:), X_hist(4,:), 'LineWidth',1.8, 'Color',[0.95 0.6 0.1], 'DisplayName','Traj');
plot(0,0,'go','MarkerFaceColor','g','DisplayName','Start');
plot(W(:,1), W(:,2), 'k--','HandleVisibility','off');
plot(W(:,1), W(:,2), 'ks', 'MarkerFaceColor','y', 'DisplayName','Waypoints');
xlabel('X [m]'); ylabel('Y [m]'); title('Lintasan kapal (LOS-NMPC, multi-waypoint)');
legend('Location','best');

subplot(3,1,2); hold on; grid on;
plot(t, X_hist(5,:), 'LineWidth',1.6);
ylabel('\psi [rad]'); title('Heading');

subplot(3,1,3); hold on; grid on;
stairs(t(1:end-1), U_hist, 'LineWidth',1.6);
yline(u_min,'--r'); yline(u_max,'--r');
ylabel('rudder u [rad]'); xlabel('time [s]'); title('Input rudder');

% ---- (Opsional) Animasi sederhana ----
figure('Color','w'); hold on; grid on; axis equal;
title('Animasi lintasan kapal'); xlabel('X [m]'); ylabel('Y [m]');
plot(W(:,1), W(:,2), 'ks-', 'MarkerFaceColor','y');
ship = plot(NaN,NaN,'bo','MarkerFaceColor','b');
trail= plot(NaN,NaN,'-','LineWidth',1.8,'Color',[0.95 0.6 0.1]);
xlim([min([0;W(:,1)])-50, max([X_hist(3,:)';W(:,1)])+50]);
ylim([min([0;W(:,2)])-50, max([X_hist(4,:)';W(:,2)])+50]);

for k = 1:size(X_hist,2)
    set(ship,'XData',X_hist(3,k),'YData',X_hist(4,k));
    set(trail,'XData',X_hist(3,1:k),'YData',X_hist(4,1:k));
    drawnow; pause(0.02);
end

end % ====== end main ======

% =====================================================================
% COST FUNCTION: LOS ke waypoint aktif sepanjang horizon (konstan)
% =====================================================================
function J = cost_LOS_waypoint(z, x0, u_prev, goal, ...
                               os_A, os_B, os_surge, dt, Np, ...
                               Qpos, Qpsi, Rdu, Ru, Qr, P, Ppsi, ...
                               du_min, du_max, W_du, ...
                               r_min, r_max, W_r, ...
                               u_min, u_max)
% z      : variabel keputusan (u di-squash dari z)
% x0     : state awal
% goal   : waypoint aktif [Xg; Yg]
% bobot  : sesuai definisi di atas
u_seq = z2u(z, u_min, u_max);
x = x0;  J = 0;

for i = 1:Np
    u = u_seq(i);

    % Prediksi satu langkah
    x_next = rk4(@(xx,uu) ship_dyn(xx,uu,os_A,os_B,os_surge), x, u, dt);
    x_next(5) = wrapToPi_local(x_next(5));

    % Error posisi & heading LOS
    e_xy   = goal - [x_next(3); x_next(4)];  % [Xg - X; Yg - Y]
    psi_des= atan2(e_xy(2), e_xy(1));        % LOS heading
    e_psi  = wrapToPi_local(psi_des - x_next(5));

    % Δu penalti + soft bounds
    if i==1, du = u - u_prev; else, du = u - u_seq(i-1); end
    du_pen = 0;
    if du < du_min, du_pen = du_pen + W_du*(du_min - du)^2; end
    if du > du_max, du_pen = du_pen + W_du*(du - du_max)^2; end

    % yaw-rate r soft bounds (x_next(2) adalah r)
    r = x_next(2);
    r_pen = 0;
    if r < r_min, r_pen = r_pen + W_r*(r_min - r)^2; end
    if r > r_max, r_pen = r_pen + W_r*(r - r_max)^2; end

    % Stage cost
    J = J ...
      + e_xy.'*Qpos*e_xy ...
      + Qpsi*(e_psi^2) ...
      + Rdu*(du^2) ...
      + Ru*(u^2) ...
      + Qr*(r^2) ...
      + du_pen + r_pen;

    x = x_next; % shift state
end

% Terminal cost (LOS juga)
eT_xy   = goal - [x(3); x(4)];
psi_des_T = atan2(eT_xy(2), eT_xy(1));
eT_psi  = wrapToPi_local(psi_des_T - x(5));
J = J + eT_xy.'*P*eT_xy + Ppsi*(eT_psi^2);
end

% =====================================================================
% DINAMIKA KAPAL (kontinu) — posisi dari geometri, lateral-yaw dari os_A,os_B
% =====================================================================
function xdot = ship_dyn(x, u, os_A, os_B, os_surge)
% x = [v; r; X; Y; psi]
v  = x(1); r = x(2); psi = x(5);
vr = os_A*[v; r] + os_B*u;   % [vdot; rdot]
vdot = vr(1);
rdot = vr(2);

xdot = [ vdot; ...
         rdot; ...
         os_surge*cos(psi) - v*sin(psi); ...
         os_surge*sin(psi) + v*cos(psi); ...
         r ];
end

% =====================================================================
% RK4 integrator
% =====================================================================
function x_next = rk4(f, x, u, h)
k1 = f(x, u);
k2 = f(x + 0.5*h*k1, u);
k3 = f(x + 0.5*h*k2, u);
k4 = f(x + h*k3, u);
x_next = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end

% =====================================================================
% z -> u (hard bound) via tanh (selalu dalam [u_min, u_max])
% =====================================================================
function u = z2u(z, u_min, u_max)
u = u_min + (u_max - u_min)*(tanh(z) + 1)/2;
end

% =====================================================================
% Wrap sudut ke [-pi,pi]
% =====================================================================
function ang = wrapToPi_local(ang)
ang = mod(ang + pi, 2*pi) - pi;
end
