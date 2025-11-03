%% ==========================================================
%  PERBANDINGAN: ODE45 vs Chebyshev Collocation (CGL)
%  Model kapal taklinier (v', r', psi', x', y')
%  ----------------------------------------------------------
clear; clc; close all;

%% ===== Parameter & Horizon
u0_surge = 2.0;      % u0' (m/s) - kecepatan surge konstan
T        = 10.0;     % panjang horizon (detik)
N        = 55;        % orde Chebyshev (disarankan 4-8 untuk awal)
s0       = [0; 0; 0; 0; 0];  % [v' r' psi' x' y'] awal

% Kontrol (kemudi) step kecil 5 derajat mulai t>=T/4
u_step_rad = deg2rad(5);
u_fun = @(t) (t>=T/4).*u_step_rad;

%% ====== SIMULASI 1: Integrasi Biasa (ODE45)
odefun = @(t,s) f_ship(s, u_fun(t), u0_surge);
opts_ode = odeset('RelTol',1e-8,'AbsTol',1e-9);
[t_ode, S_ode] = ode45(odefun, [0 T], s0, opts_ode);

%% ====== SIMULASI 2: Chebyshev–Gauss–Lobatto Collocation
[tau, D]  = cgl_nodes_and_D(N);          % node & matriks D
t_nodes   = (T/2)*(tau+1);               % map ke domain waktu [0,T]

% Tebakan awal = hasil ODE45 yang diinterpolasi
S0_guess = interp1(t_ode, S_ode, t_nodes, 'pchip','extrap');

% Bangun fungsi residual kolokasi
resfun = @(vec) residual_collocation(vec, D, T, t_nodes, u_fun, u0_surge, N);
vec0 = S0_guess(:);

% Solver nonlinear: coba fsolve dulu, jika gagal, fallback ke fminsearch
try
    opts = optimoptions(@fsolve,'Display','iter',...
        'Algorithm','trust-region-dogleg',...
        'MaxIterations',1000,'FunctionTolerance',1e-6);
    [vec_sol,~,exitflag] = fsolve(resfun,vec0,opts);
    if exitflag <= 0
        warning('fsolve tidak konvergen, lanjut pakai fminsearch...');
        objfun = @(x) norm(resfun(x))^2;
        vec_sol = fminsearch(objfun,vec0);
    end
catch
    % Jika tidak punya Optimization Toolbox
    objfun = @(x) norm(resfun(x))^2;
    vec_sol = fminsearch(objfun,vec0);
end

S_cheb = reshape(vec_sol,[N+1,5]);

%% ====== Plot Perbandingan
names = ["v' (m/s)","r' (rad/s)","\psi' (rad)","x' (m)","y' (m)"];
figure('Color','w','Position',[100 100 1100 700]);
for j=1:5
    subplot(3,2,j);
    plot(t_ode,  S_ode(:,j),'LineWidth',1.8); hold on;
    plot(t_nodes,S_cheb(:,j),'o-','LineWidth',1.2);
    grid on; xlabel('t (s)'); ylabel(names(j));
    legend('ODE45','Chebyshev CGL','Location','best');
    title(names(j));
end
sgtitle(sprintf('Perbandingan ODE45 vs Chebyshev CGL (T=%.1f s, N=%d)',T,N));

%% ====================== FUNGSI-FUNGSI ======================

function ds = f_ship(s, u, u0)
% Model kapal taklinier (5 state)
v   = s(1); r = s(2); psi = s(3);
dv   = -0.6174*v - 0.1036*r + 0.01*u;
dr   = -5.0967*v - 3.4047*r + u;
dpsi = r;
dx   = u0*cos(psi) - v*sin(psi);
dy   = u0*sin(psi) + v*cos(psi);
ds = [dv; dr; dpsi; dx; dy];
end

function [tau, D] = cgl_nodes_and_D(N)
% Titik Chebyshev–Gauss–Lobatto dan matriks D
k   = (0:N).'; tau = cos(pi*k/N);
c = ones(N+1,1); c([1 end]) = 2;
D = zeros(N+1);
for i=1:N+1
    for j=1:N+1
        if i~=j
            D(i,j) = (c(i)/c(j))*(-1)^(i+j)/(tau(i)-tau(j));
        end
    end
end
D(1,1)     =  (2*N^2+1)/6;
D(N+1,N+1) = -(2*N^2+1)/6;
for i=2:N
    D(i,i) = -tau(i)/(2*(1-tau(i)^2));
end
end

function F = residual_collocation(vec,D,T,t_nodes,u_fun,u0,N)
% Residual kolokasi: sum_i D_{ki} S_i - (T/2)*f(S_k,u_k)=0
S = reshape(vec,[N+1,5]);
F = zeros((N+1)*5,1);
for k=1:N+1
    Sk = S(k,:).'; uk = u_fun(t_nodes(k));
    fk = f_ship(Sk, uk, u0);
    lhs = D(k,:)*S; rhs = (T/2)*fk.';
    F(5*(k-1)+(1:5)) = (lhs - rhs).';
end
end
