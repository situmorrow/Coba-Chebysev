%% ==========================================================
%  PERBANDINGAN: RK4 vs Chebyshev Collocation (CGL)
%  Model kapal taklinier (v', r', psi', x', y')
%  ----------------------------------------------------------
clear; clc; close all;

%% ===== Parameter & Horizon
u0_surge = 2.0;      % u0' (m/s) - kecepatan surge konstan
T        = 10.0;     % panjang horizon (detik)
N        = 30;       % orde Chebyshev (CGL), default 10 (bisa 4, 8, 12, ...)
dt_rk4   = 0.01;     % step integrasi RK4 (detik)

% Kondisi awal
s0 = [0; 0; 0; 0; 0];  % [v' r' psi' x' y']^T

% Kontrol sebagai fungsi waktu (dipakai di RK4 & Chebyshev)
% Contoh: step kecil 5 derajat pada t>=T/4 (konversi ke rad)
u_step_rad = deg2rad(5);
u_fun = @(t) (t>=T/4).*u_step_rad;   % bisa diganti sesuai kebutuhan

%% ====== SIMULASI 1: Integrasi Biasa (RK4)
[t_rk, S_rk] = simulate_rk4(@(t,s) f_ship(s, u_fun(t), u0_surge), 0, T, s0, dt_rk4);

%% ====== SIMULASI 2: Chebyshev–Gauss–Lobatto Collocation
% Bentuk node CGL dan matriks diferensiasi
[tau, D]  = cgl_nodes_and_D(N);          % tau in [-1,1], size N+1
t_nodes   = (T/2)*(tau+1);               % peta tau -> t in [0,T]
% Bangun residual kolokasi dan pecahkan dengan fsolve
% (gunakan solusi RK4 sebagai initial guess agar konvergen bagus)
S0_guess  = interp1(t_rk, S_rk, t_nodes, 'pchip','extrap');   % (N+1) x 5

opts = optimoptions(@fsolve, ...
    'Algorithm','trust-region-dogleg', ... % lebih stabil
    'Display','iter', ...
    'MaxIterations', 1000, ...
    'FunctionTolerance', 1e-6, ...
    'StepTolerance', 1e-6);

vec0 = S0_guess(:);  % vectorize initial guess
resfun = @(vec) residual_collocation(vec, D, T, t_nodes, u_fun, u0_surge, N);
[vec_sol, ~, exitflag] = fsolve(resfun, vec0, opts);
if exitflag <= 0
    warning('fsolve tidak konvergen sepenuhnya, gunakan solusi iterasi terakhir.');
end

S_cheb = reshape(vec_sol, [N+1, 5]);   % urutan kolom: [v' r' psi' x' y']

%% ====== Plot Perbandingan
names = ["v' (m/s)","r' (rad/s)","\psi' (rad)","x' (m)","y' (m)"];
figure('Color','w','Position',[100 100 1100 700]);
for j=1:5
    subplot(3,2,j);
    plot(t_rk,  S_rk(:,j), 'LineWidth',1.6); hold on;
    plot(t_nodes, S_cheb(:,j), 'o-','LineWidth',1.2);
    grid on; xlabel('t (s)'); ylabel(names(j));
    legend('RK4','Chebyshev CGL','Location','best'); title(names(j));
end
sgtitle(sprintf('Perbandingan RK4 vs Chebyshev CGL  (T=%.1f s, N=%d)',T,N));

%% ====== (Opsional) Rekonstruksi kurva halus dari Chebyshev
% Jika ingin kurva halus dari Chebyshev: evaluasi interpolasi Lagrange.
% contoh untuk psi':
% tt = linspace(0,T,1000);
% tauq = (2*tt/T)-1;
% L = lagrange_basis_matrix(tau, tauq);     % (N+1)xM
% psi_interp = (S_cheb(:,3).')*L;           % 1xM
% figure; plot(t_rk,S_rk(:,3)); hold on; plot(tt,psi_interp);

%% ====================== FUNGSI-FUNGSI ======================

function ds = f_ship(s, u, u0)
% Model kapal taklinier (turunannya)
% s = [v; r; psi; x; y]
v   = s(1);
r   = s(2);
psi = s(3);

dv   = -0.6174*v - 0.1036*r + 0.01*u;
dr   = -5.0967*v - 3.4047*r + u;
dpsi = r;
dx   = u0*cos(psi) - v*sin(psi);
dy   = u0*sin(psi) + v*cos(psi);

ds = [dv; dr; dpsi; dx; dy];
end

function [t, S] = simulate_rk4(f, t0, tf, s0, dt)
% Integrasi RK4 standar
t = (t0:dt:tf).';
S = zeros(numel(t), numel(s0));
S(1,:) = s0.';
for k=1:numel(t)-1
    h  = dt;
    tk = t(k);
    sk = S(k,:).';
    k1 = f(tk,        sk);
    k2 = f(tk+h/2.0,  sk + (h/2.0)*k1);
    k3 = f(tk+h/2.0,  sk + (h/2.0)*k2);
    k4 = f(tk+h,      sk + h*k3);
    S(k+1,:)= (sk + (h/6)*(k1 + 2*k2 + 2*k3 + k4)).';
end
end

function [tau, D] = cgl_nodes_and_D(N)
% Node Chebyshev–Gauss–Lobatto dan matriks diferensiasi
% tau_k = cos(pi*k/N), k=0..N
k   = (0:N).';
tau = cos(pi*k/N);

c         = ones(N+1,1);
c(1)      = 2; 
c(end)    = 2;
D         = zeros(N+1);
for i=1:N+1
    for j=1:N+1
        if i~=j
            D(i,j) = (c(i)/c(j)) * (-1)^(i+j) / (tau(i)-tau(j));
        end
    end
end
% diagonal
D(1,1)     =  (2*N^2+1)/6;
D(N+1,N+1) = -(2*N^2+1)/6;
for i=2:N
    D(i,i) = -tau(i)/(2*(1-tau(i)^2));
end
end

function F = residual_collocation(vec, D, T, t_nodes, u_fun, u0, N)
% Residual kolokasi Chebyshev: sum_i D_{ki} S_i - (T/2) f(S_k, u_k) = 0
% vec: (N+1)*5 vektor [v0..vN, r0..rN, psi0..psiN, x0..xN, y0..yN]
S = reshape(vec, [N+1, 5]);        % baris k=1..N+1
F = zeros((N+1)*5,1);
for k=1:N+1
    Sk = S(k,:).';                % state di titik k
    uk = u_fun(t_nodes(k));       % kontrol di titik k (diberi)
    fk = f_ship(Sk, uk, u0);      % f(S_k,u_k)
    % LHS: D(k,:)*S(:,j) untuk j=1..5
    lhs = (D(k,:)*S);             % 1x5
    rhs = (T/2)*fk.';             % 1x5
    F(5*(k-1)+(1:5)) = (lhs - rhs).';  % simpan residual komponen
end
end

function L = lagrange_basis_matrix(tau_nodes, tauq)
% Matriks basis Lagrange, L(i,q) = phi_i(tauq(q)), i=0..N
N  = numel(tau_nodes)-1;
M  = numel(tauq);
L  = ones(N+1,M);
for i=1:N+1
    for j=1:N+1
        if j~=i
            L(i,:) = L(i,:).*( (tauq - tau_nodes(j))./(tau_nodes(i)-tau_nodes(j)) );
        end
    end
end
end