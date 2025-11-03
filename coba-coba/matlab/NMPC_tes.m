clc; clear; close all;

%% Parameter
dt = 1; % time sampling
Np = 60; % prediction horizon

% input
tactical_diameter = 948;
r_max = 0.0932; r_min = -r_max;
u_max = deg2rad(35); u_min = -u_max;
rrot_max = deg2rad(5); rrot_min = -rrot_max; 

%% Corvette SIGMA class
% TA mas rendy hlm 27
Lpp = 101.07; % Lpp
B = 14; % breadth - meter
T = 3.7; % design wraught (DWL) - meter
m = 2423 * 1e3; % displacement - kg
os_surge = 15.4; % surge velocity - m/s
CB = 0.65; % block coefficient
xG = 5.25; % center of gravity
rho = 1024; % sea water density - kg/m^3
Adelta = 5.7224; % rudder area
gyration = 0.156*Lpp; % radius of gyration (0.15Lpp < r < 0.3Lpp)

%% Hydrodynamics Coefficient
% rumus dari hlm 10 dan hasil hlm 28
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

% hlm 11
nd_u = 1;
% hlm 9
nd_m = m/(0.5*rho*Lpp^3);
nd_xG=xG/Lpp;

%% Mathematical model 
M=[nd_m-Yvdot nd_m*nd_xG-Yrdot;
   nd_m*nd_xG-Nvdot Iz-Nrdot];
N=[-Yv nd_m*nd_u-Yr;
   -Nv nd_m*nd_xG*nd_u-Nr];

model_A = -inv(M)*N; 
a11 = model_A(1,1); a12 = model_A(1,2);
a21 = model_A(2,1); a22 = model_A(2,2);

b = [0.01;1];
b11 = b(1); b12 = b(2);
model_B = [b11; b12/Lpp]; % dari penelitian Fadia

% xdot = Ax + Bu
%os_U = sqrt(os_surge^2 + v^2); % total speed
os_U = os_surge;
os_A = os_U*[a11 a12*Lpp; a21/Lpp a22]/Lpp;
os_B = os_U^2*model_B/Lpp;

%% Bobot biaya 
w_position = 1e-4;
w_control = 1e-2;
w_orientation = 0.1;

%% Inisialisasi MPC
x = [0;0;0;0;pi/2]; % [v;r;x;y;psi]
ref = [500; 2000; 0]; % referensi [x_ref;y_ref;psi_ref]
axis_lim = [-200 3600 -200 2200];

u_prev = 0;
z_opt = zeros(Np+1,1);

max_iter = 300;
dist_tol = 10;

X_hist = x;
U_hist = [];

% Opsi Nelder–Mead
opts = optimset('Display','off','MaxFunEvals',5e4,'MaxIter',5e4,'TolX',1e-6,'TolFun',1e-6);

fprintf('RUN NMPC');

while norm([x(3):x(4)]-ref(1:2)) > dist_tol && mpc_k < max_iter

    % costfunction untuk horizon sekarang
    costfun = @(z) nmpc_cost(z, x0, u_prev, ref, ...
                            os_A, os_B, os_surge, dt, Np, ...
                            w_position, w_control, w_orientation, ...
                            r_min, r_max, rrot_min,rrot_max, ...
                            u_min, u_max);

    % Optimasi
    % warm start: geser solusi sebelumnya
    if mpc_k>0, z_opt = [z_op(2:end); z_opt(end)]; end
    [z_opt, ~] = fminsearch(costfun, z_opt, opts);

    % ===== Terapkan kontrol pertama =====
    u_seq = z2u(z_opt, u_min, u_max);
    u = u_seq(1);
    U_hist(end+1,1) = u;

    % ===== Simulasi 1 langkah (Euler) =====
    xdot = ship_model_cont(x, u, os_A, os_B, os_surge);
    x = rk4(@(ss,uu) xdot, x, u_seq(k), dt);               % Euler fwd
    x(5) = wrapToPi_local(x(5));   % wrap psi
    X_hist(:,end+1) = x;

    % ===== Simpan prediksi terakhir (opsional untuk plot) =====
    pred_hist_last = rollout_pred(x - dt*xdot, z_opt, os_A,os_B,os_surge, dt, Np); %#ok<NASGU>

    % ===== Update =====
    u_prev = u;
    mpc_k  = mpc_k + 1;

end

%% Cost Function
function J = nmpc_cost(z, x, u_prev, ref, ...
                            os_A, os_B, os_surge, dt, Np, ...
                            w_position, w_control, w_orientation, ...
                            r_min, r_max, rrot_min,rrot_max, ...
                            u_min, u_max)

    u_seq = z2u(z, u_min, u_max);
    x = x0;  J = 0;
    wrap = @(a) atan2(sin(a), cos(a))
    
    for k = 1:Np+1
        u = u_seq(k);
        xdot = ship_model_cont(x, u, os_A, os_B, os_surge);
        x_next = rk4(@(ss,uu) xdot, x, u_seq(k), dt)
        x_next(5) = wrapToPi_local(x_next(5));

        if k <= Np
            e_xy = [ref(1)-x_next(3); ref(2)-x_next(4)];
            psi_des = atan2(e_xy(2), e_xy(1));
            e_psi = wrap(psi_des-x_next(5));

            J = J + w_pos*(e_xy.'*e_xy) + w_psi*(e_psi^2) + w_u*(u^2);
            
            % Δu bounds (soft)
            if k==1, du = u - u_prev; else, du=u-u_seq(k-1); end
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
    eT_psi  = wrap(psi_des_T - x(5));
    J = J + 10*w_pos*(eT_xy.'*eT_xy) + 10*w_psi*(eT_psi^2);
end
%% Model Kontinu
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
%% Runge Kutta Orde 4
function x_rk = rk4(f, s, u, h)
    k1 = f(s, u);
    k2 = f(s + 0.5*h*k1, u);
    k3 = f(s + 0.5*h*k2, u);
    k4 = f(s + h*k3, u);
    s_next = s + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end

%% wrap sudut
function ang = wrapToPi_local(ang)
    ang = mod(ang + pi, 2*pi) - pi;
end

%% util: z -> u
function u = z2u(z, u_min, u_max)
    if isfinite(u_min) && isfinite(u_max)
        u = u_min + (u_max - u_min) * (tanh(z) + 1)/2;
    else
        u = z; % jika dipanggil hanya untuk tracing
    end
end
%% UTIL: prediksi penuh (opsional plot)
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
