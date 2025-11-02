function nmpc_ship_fmincon()
clc; clear; close all;

%% ===== Parameter dasar & horizon =====
dt = 0.5;                 % time step
Np = 30;                % prediction horizon (steps)

% Batas input & rate (rudder)
r_max = 0.0932; r_min = -r_max;                  % yaw-rate bounds
u_max = deg2rad(35); u_min = -u_max;             % rudder bounds
rrot_max = deg2rad(5); rrot_min = -rrot_max;     % |Δu| bounds per step

% ----- State: [v;r;x;y;psi] -----
% (parameter & linearisasi ringkas—sesuaikan dengan modelmu)
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
Ir = (m*(gyration^2))/(0.5*rho*Lpp^5);
Iz = (m*(xG^2))/(0.5*rho*Lpp^5)+Ir;
nd_u = 1; nd_m = m/(0.5*rho*Lpp^3); nd_xG = xG/Lpp;
M = [nd_m-Yvdot,          nd_m*nd_xG-Yrdot; ...
     nd_m*nd_xG-Nvdot,    Iz-Nrdot];
Nmat = [-Yv,              nd_m*nd_u-Yr; ...
        -Nv,              nd_m*nd_xG*nd_u-Nr];
Acont = -(M\Nmat);
a11=Acont(1,1); a12=Acont(1,2); a21=Acont(2,1); a22=Acont(2,2);
b  = [0.01;1];
model_B = [b(1); b(2)/Lpp];

os_U = os_surge;
os_A = os_U*[a11 a12*Lpp; a21/Lpp a22]/Lpp;
os_B = os_U^2*model_B/Lpp;

% Bobot biaya
w_position     = 1e-4;
w_orientation  = 20;
w_control      = 1e-2;
w_du           = 1e-1;

%% ===== Inisialisasi =====
x = [0;0;0;0;pi/2];            % [v;r;x;y;psi] awal
ref = [500;1000;0];            % [x_ref, y_ref, psi_ref]
u_prev = 0;                     % u(k-1) untuk batas Δu pada langkah pertama

X_hist = x; U_hist = [];
dist_tol = 10; max_iter = 10000; kloop = 0;

% Variabel keputusan: u_seq (Np+1 x 1)
u0 = zeros(Np+1,1);                        % tebakan awal
lb = u_min*ones(Np+1,1);                   % batas kemudi
ub = u_max*ones(Np+1,1);

% Kendala linear Δu (A u <= b)
[Adu, bdu] = build_rate_constraints(Np+1, rrot_min, rrot_max, u_prev);

% Opsi fmincon
opts = optimoptions('fmincon', ...
    'Algorithm','sqp', ...
    'MaxIterations', 800, ...
    'MaxFunctionEvaluations', 1.5e5, ...
    'StepTolerance', 1e-7, ...
    'OptimalityTolerance', 5e-6, ...
    'Display','iter');

fprintf('RUN NMPC (fmincon, single-shooting)\n');

while norm([x(3);x(4)] - ref(1:2)) > dist_tol && kloop < max_iter

    % ---- Cost function (tanpa penalti kendala) ----
    costfun = @(u) nmpc_cost_u(u, x, ref, os_A, os_B, os_surge, dt, Np, ...
                           w_position, w_control, w_orientation, w_du, u_prev);

    % ---- Kendala nonlinier: r_k in [r_min, r_max] sepanjang horizon ----
    nonlcon = @(u) yawrate_bounds(u, x, os_A, os_B, os_surge, dt, Np, r_min, r_max);

    % ---- Optimasi ----
    if kloop>0, u0 = [u_opt(2:end); u_opt(end)]; end % warm-start shift
    u_opt = fmincon(costfun, u0, Adu, bdu, [], [], lb, ub, nonlcon, opts);

    % Terapkan kontrol pertama
    u = u_opt(1);
    U_hist(end+1,1) = u;

    % Simulasi 1 langkah (Euler)
    xdot = ship_model_cont(x, u, os_A, os_B, os_surge);
    x = x + dt*xdot; x(5) = wrapToPi_local(x(5));
    X_hist(:,end+1) = x;

    % Update kendala Δu untuk iterasi berikut (u_prev berubah)
    u_prev = u;
    [Adu, bdu] = build_rate_constraints(Np+1, rrot_min, rrot_max, u_prev);

    kloop = kloop + 1;
end

%% ===== Plot =====
t = 0:dt:(size(X_hist,2)-1)*dt;
figure('Color','w');
subplot(3,1,1); hold on; grid on; axis equal;
plot(X_hist(3,:), X_hist(4,:), 'LineWidth',1.5);
plot(0,0,'go','MarkerFaceColor','g'); plot(ref(1),ref(2),'r*','MarkerSize',10);
xlabel('x [m]'); ylabel('y [m]'); title('Lintasan');

subplot(3,1,2); hold on; grid on;
plot(t, X_hist(5,:), 'LineWidth',1.5); yline(ref(3),'--r');
ylabel('\psi [rad]'); title('Heading');

subplot(3,1,3); hold on; grid on;
stairs(t(1:end-1), U_hist, 'LineWidth',1.5);
yline(u_min,'--r'); yline(u_max,'--r');
ylabel('rudder [rad]'); xlabel('time [s]');

end

%% ===== Cost: jumlah error LOS + effort u =====
function J = nmpc_cost_u(u_seq, x0, ref, A, B, U, dt, Np, ...
                         w_pos, w_u, w_psi, w_du, u_prev)
% Cost NMPC berbasis LOS + penalti Δu
x = x0; 
J = 0;
up = u_prev;    % u_{k-1} untuk k=1

for k = 1:Np
    u = u_seq(k);

    % rollout 1 langkah
    x = x + dt*ship_model_cont(x, u, A, B, U);
    x(5) = wrapToPi_local(x(5));

    % === LOS angle (arah ke target) ===
    psi_des = atan2(ref(2)-x(4), ref(1)-x(3));
    e_psi   = wrapToPi_local(psi_des - x(5));

    % error posisi
    e_xy = [ref(1)-x(3); ref(2)-x(4)];

    % penalti Δu
    du = u - up;

    % akumulasi biaya
    J = J ...
        + w_pos*(e_xy.'*e_xy) ...
        + w_psi*(e_psi^2) ...
        + w_u*(u^2) ...
        + w_du*(du^2);

    up = u; % update u_{k-1}
end

% === Terminal cost lebih berat ===
eT_xy   = [ref(1)-x(3); ref(2)-x(4)];
psi_des = atan2(ref(2)-x(4), ref(1)-x(3));
eT_psi  = wrapToPi_local(psi_des - x(5));
J = J + 10*w_pos*(eT_xy.'*eT_xy) + 10*w_psi*(eT_psi^2);
end


%% ===== Kendala nonlinier: r_k in [r_min, r_max] untuk semua k =====
function [c, ceq] = yawrate_bounds(u_seq, x0, A, B, U, dt, Np, rmin, rmax)
x = x0; r_list = zeros(Np,1);
for k = 1:Np
    u = u_seq(k);
    x = x + dt*ship_model_cont(x, u, A, B, U);
    x(5) = wrapToPi_local(x(5));
    r_list(k) = x(2);
end
% c<=0  →  [rmin - r_k; r_k - rmax] <= 0
c = [rmin - r_list; r_list - rmax];
ceq = [];
end

%% ===== Kendala linear Δu: bangun A,b untuk semua langkah =====
function [A, b] = build_rate_constraints(N, rrot_min, rrot_max, u_prev)
% N = Np+1 panjang u_seq
% Buat dua set:  (u_k - u_{k-1}) <= rrot_max  dan  -(u_k - u_{k-1}) <= -rrot_min
A = zeros(2*N, N); b = zeros(2*N,1);
row = 0;

% Upper:  u1 - u_prev <= rrot_max
row=row+1; A(row,1) = 1; b(row) = rrot_max + u_prev;
% Lower: -(u1 - u_prev) <= -rrot_min  →  -u1 <= -rrot_min - u_prev
row=row+1; A(row,1) = -1; b(row) = -rrot_min - u_prev;

% Untuk k=2..N:  uk - u_{k-1} <= rrot_max ;  -(uk - u_{k-1}) <= -rrot_min
for k=2:N
    row=row+1; A(row,k) = 1;  A(row,k-1) = -1; b(row) = rrot_max;
    row=row+1; A(row,k) = -1; A(row,k-1) =  1; b(row) = -rrot_min;
end

% Pangkas baris berlebih (jaga-jaga)
A = A(1:row,:); b = b(1:row);
end

%% ===== Dinamika kontinu terlinierisasi (format sama dengan punyamu) =====
function xdot = ship_model_cont(x, u, os_A, os_B, os_surge)
% x = [v;r;x;y;psi]
v = x(1); r = x(2); psi = x(5);
dyn = os_A*[v; r] + os_B*u;
vdot = dyn(1); rdot = dyn(2);
xdot = [ vdot;
         rdot;
         os_surge*cos(psi) - v*sin(psi);
         os_surge*sin(psi) + v*cos(psi);
         r ];
end

function ang = wrapToPi_local(ang)
ang = mod(ang + pi, 2*pi) - pi;
end
