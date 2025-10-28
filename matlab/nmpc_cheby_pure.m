function nmpc_cheby_pure()
% =====================================================================
% NMPC Chebyshev Pseudospectral — PURE MATLAB (tanpa toolbox)
% - Domain tau ∈ [-1,1], titik CGL, matriks diferensiasi D
% - Kendala kolokasi: D*S = (T/2) f(S,u) (dipaksakan via penalti besar)
% - Biaya: tracking [psi,x,y] + penalti u_dot (pakai D*u)
% - Rudder u dibatasi keras via tanh; u_dot dibatasi lunak (penalti)
% =====================================================================

clc; clear; close all;

%% ====== Parameter model (ringkas) ======
Lpp = 101.07; B = 14; Tm = 3.7; m = 2423e3; Uc = 15.4; CB = 0.65;  % <--- Uc (bukan U!)
xG = 5.25; rho = 1024; Adelta = 5.7224; gyr = 0.156*Lpp; %#ok<NASGU>

Yvdot = -((1+0.16*CB*(B/Tm)-5.1*(B/Lpp)^2)*pi*(Tm/Lpp)^2);
Yrdot = -((0.67*(B/Lpp)-0.0033*(B/Tm)^2)*pi*(Tm/Lpp)^2);
Nvdot = -((1.1*(B/Lpp)-0.041*(B/Tm))*pi*(Tm/Lpp)^2);
Nrdot = -(((1/12)+0.017*(CB*B/Tm)-0.33*(B/Lpp))*pi*(Tm/Lpp)^2);
Yv    = -((1+0.4*(CB*B/Tm))*pi*(Tm/Lpp)^2);
Yr    = -((-0.5+2.2*(B/Lpp)-0.08*(B/Tm))*pi*(Tm/Lpp)^2);
Nv    = -((0.5+2.4*(Tm/Lpp))*pi*(Tm/Lpp)^2);
Nr    = -((0.25+0.039*(B/Tm)-0.56*(B/Lpp))*pi*(Tm/Lpp)^2);

Ir = (m*(gyr^2))/(0.5*rho*Lpp^5);
Iz = (m*(xG^2))/(0.5*rho*Lpp^5)+Ir;

nd_u = 1; nd_m = m/(0.5*rho*Lpp^3); nd_xG = xG/Lpp;

M = [nd_m-Yvdot,           nd_m*nd_xG-Yrdot; ...
     nd_m*nd_xG-Nvdot,     Iz-Nrdot];
Nmat = [-Yv,               nd_m*nd_u-Yr; ...
        -Nv,               nd_m*nd_xG*nd_u-Nr];
A = -(M\Nmat);  a11=A(1,1); a12=A(1,2); a21=A(2,1); a22=A(2,2);

b = [0.01; 1];
Bv = [b(1); b(2)/Lpp];

os_A = Uc*[a11 a12*Lpp; a21/Lpp a22]/Lpp;
os_B = Uc^2*Bv/Lpp;

%% ====== Pseudospectral grid (CGL) ======
N   = 12;             % derajat (node = N+1)
T_h = 12.0;           % panjang horizon [s]
[tau, D] = cheb_CGL_diff(N);
w = trapezoid_weights_on_tau(N);  % bobot integrasi sederhana di tau

% h = C*S, urutan keluaran [psi; x; y] dari S=[v;r;psi;x;y]
Cout = [0 0 1 0 0;
        0 0 0 1 0;
        0 0 0 0 1];

%% ====== Bobot biaya & batas ======
Q = diag([6, 12, 12]);   % untuk [psi, x, y]
R = 0.05;                % penalti u_dot
Wdyn = 1e5;              % penalti kolokasi
Wic  = 1e6;              % penalti initial condition
Wu_soft = 0;             % (opsional) penalti u^2 kecil

u_max = deg2rad(35);  u_min = -u_max;
udot_max = deg2rad(6); udot_min = -udot_max;
Wratedot = 1e4;

% Target posisi (LOS untuk heading)
ref_pos = [500; 1000];

%% ====== Loop NMPC ======
dt_sim = 1.0;  Tsim = 120;  Ksim = round(Tsim/dt_sim);

S_now = [0; 0; pi/2; 0; 0];     % [v;r;psi;x;y]

% Keputusan awal
S_guess = repmat(S_now,1,N+1);
U_guess = zeros(1,N+1);

z = pack_decision(S_guess, atanh( map_u_to_unit(U_guess,u_min,u_max) ));

S_log = S_now; U_log = [];  % untuk plot

opts = optimset('Display','off','MaxFunEvals',6e4,'MaxIter',6e4,'TolX',1e-6,'TolFun',1e-6);

for k = 1:Ksim
    costfun = @(zz) cost_cheby(zz, S_now, ref_pos, ...
        os_A, os_B, Uc, N, T_h, tau, D, w, Cout, Q, R, Wdyn, Wic, ...
        u_min, u_max, udot_min, udot_max, Wratedot, Wu_soft);

    % warm start
    z = shift_warmstart(z, N);

    % optimasi
    z = fminsearch(costfun, z, opts);

    % kontrol pertama & propagate plant
    [S_nodes, U_nodes_z] = unpack_decision(z, N);
    u_nodes = squash_u(U_nodes_z, u_min, u_max);     % <--- pakai u_nodes (bukan U)
    u_apply = u_nodes(1);

    S_now = rk4(@(x,u) f_ship(x,u,os_A,os_B,Uc), S_now, u_apply, dt_sim);  % <--- Uc skalar
    S_now(3) = wrapToPi_local(S_now(3));

    % log
    S_log(:,end+1) = S_now;
    U_log(end+1,1) = u_apply;
end

%% ====== Plot ======
t = 0:dt_sim:(size(S_log,2)-1)*dt_sim;

figure('Color','w');
subplot(3,1,1); hold on; grid on; axis equal;
plot(S_log(4,:), S_log(5,:), 'LineWidth',1.8, 'Color',[0.95 0.6 0.1]);
plot(0,0,'go','MarkerFaceColor','g'); plot(ref_pos(1), ref_pos(2), 'r*','MarkerSize',10);
xlabel('x [m]'); ylabel('y [m]'); title('Lintasan (Chebyshev NMPC)');

subplot(3,1,2); hold on; grid on;
plot(t, S_log(3,:), 'LineWidth',1.6);
ylabel('\psi [rad]'); title('Heading');

subplot(3,1,3); hold on; grid on;
stairs(t(1:end-1), U_log, 'LineWidth',1.6);
yline(u_min,'--r'); yline(u_max,'--r');
ylabel('u [rad]'); xlabel('time [s]'); title('Input rudder');

end
% ===================== end main ==============================


% ===================== COST CHEBYSHEV =========================
function J = cost_cheby(z, S0, ref_pos, os_A, os_B, Uc, ...
                        N, T_h, tau, D, w, C, Q, R, Wdyn, Wic, ...
                        u_min, u_max, udot_min, udot_max, Wratedot, Wu_soft)

[S_nodes, Uz_nodes] = unpack_decision(z, N);
u_nodes = squash_u(Uz_nodes, u_min, u_max);    % u ∈ [u_min,u_max]

% Kendala kolokasi: D*S = (T/2) f(S,u)
F = zeros(size(S_nodes));  % 5 x (N+1)
for k = 1:N+1
    F(:,k) = f_ship(S_nodes(:,k), u_nodes(k), os_A, os_B, Uc);
end
res_dyn = D*S_nodes.' - (T_h/2)*F.';     % (N+1) x 5
J_dyn = Wdyn * sum(res_dyn(:).^2);

% Initial condition di node k=0
J_ic  = Wic * sum( (S_nodes(:,1) - S0).^2 );

% u_dot via D*u (di tau), lalu skala 2/T_h
udot_nodes = (2/T_h) * (D*u_nodes.');   % (N+1) x 1

% Bound u_dot (soft)
viol_udot = max(0, udot_min - udot_nodes) + max(0, udot_nodes - udot_max);
J_udot_bound = Wratedot * sum(viol_udot.^2);

% Tracking + penalti u_dot di setiap node (pakai bobot w)
J_track = 0;
for k = 1:N+1
    Sk = S_nodes(:,k);
    hk = C*Sk;                               % [psi; x; y]
    e_xy = ref_pos - hk(2:3);
    psi_des = atan2(e_xy(2), e_xy(1));       % LOS
    e = [wrapToPi_local(psi_des - hk(1)); e_xy(1); e_xy(2)];  % [e_psi; e_x; e_y]

    J_track = J_track + (T_h/2) * w(k) * ( e.'*Q*e + Wu_soft*(u_nodes(k)^2) + R*(udot_nodes(k)^2) );
end

J = J_dyn + J_ic + J_udot_bound + J_track;
end

% ===================== DINAMIKA KAPAL =======================
function xdot = f_ship(x, u, os_A, os_B, Uc)
% x = [v; r; psi; x; y]
v = x(1); r = x(2); psi = x(3);
vr = os_A*[v; r] + os_B*u;    % [vdot; rdot]
vd = vr(1); rd = vr(2);
xdot = [ vd;
         rd;
         r;
         Uc*cos(psi) - v*sin(psi);
         Uc*sin(psi) + v*cos(psi) ];
end

% ===================== RK4 INTEGRATOR =======================
function x_next = rk4(f, x, u, h)
k1 = f(x,u);
k2 = f(x + 0.5*h*k1, u);
k3 = f(x + 0.5*h*k2, u);
k4 = f(x + h*k3, u);
x_next = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end

% ===================== CGL nodes & D-matrix =================
function [tau, D] = cheb_CGL_diff(N)
k = (0:N).';
tau = cos(pi*k/N);
c = [2; ones(N-1,1); 2] .* ((-1).^k);
X = repmat(tau,1,N+1);
dX = X - X.';
D  = (c*(1./c)')./(dX + eye(N+1));
D  = D - diag(sum(D,2));
end

% ===================== Bobot trapezoidal di tau ∈ [-1,1] ===
function w = trapezoid_weights_on_tau(N)
w = ones(N+1,1)*(2/N); w(1)=w(1)/2; w(end)=w(end)/2;
end

% ===================== Packing / Unpacking =================
function z = pack_decision(S_nodes, Uz_nodes)
z = [S_nodes(:); Uz_nodes(:)];
end
function [S_nodes, Uz_nodes] = unpack_decision(z, N)
nx = 5; m = 1;
lenS = nx*(N+1);
S_nodes = reshape(z(1:lenS), nx, N+1);
Uz_nodes = reshape(z(lenS+1:end), m, N+1);
end

% ========== Squash & mapping u ===========
function u_nodes = squash_u(Uz_nodes, umin, umax)
u_nodes = umin + (umax - umin)*(tanh(Uz_nodes) + 1)/2;
end
function uu = map_u_to_unit(U_nodes, umin, umax)
uu = 2*(U_nodes - umin)/(umax-umin) - 1;
uu = min(max(uu, -0.999), 0.999);
end

% ========== Warm start: shift node ===========
function z_new = shift_warmstart(z, N)
[S,U] = unpack_decision(z,N);
S = [S(:,2:end), S(:,end)];
U = [U(:,2:end), U(:,end)];
z_new = pack_decision(S,U);
end

% ========== wrap sudut (tanpa toolbox) ===========
function ang = wrapToPi_local(ang)
ang = mod(ang + pi, 2*pi) - pi;
end
