%% Perbandingan Chebyshev Pseudospectral vs ODE45
% ODE: x' = x, x(0)=1, t in [0,1]

clear; clc; close all;

% ----- Problem setup
t0 = 0; tf = 10; T = tf - t0;
x0 = 1;

% ====== 1) Chebyshev–Gauss–Lobatto nodes & D-matrix ======
N = 20;                    % derajat polinomial (jumlah node = N+1)
[taus, D] = chebDiffMat(N);% taus: [1 ... -1] (urutan CGL standar)
t_nodes = (T/2)*(taus + 1) + t0;   % mapping [-1,1] -> [t0,tf]

% ----- 2) Kolokasi global: (2/T) D X - X = 0, ganti baris BC di τ=-1
A = (2/T)*D - eye(N+1);
b = zeros(N+1,1);

% indeks τ=-1 adalah i = N (baris terakhir pada urutan CGL klasik)
A(end,:) = 0; 
A(end,end) = 1;
b(end) = x0;

Xc = A \ b;                % nilai x pada node CGL

% ====== 3) ODE45 sebagai pembanding
odefun = @(t,x) x;
opts = odeset('RelTol',1e-10,'AbsTol',1e-12);
[t45, x45] = ode45(odefun, [t0 tf], x0, opts);

% ====== 4) Solusi eksak & evaluasi error
x_exact_nodes = exp(t_nodes);
max_err_cheb  = max(abs(Xc - x_exact_nodes));
max_err_ode45 = max(abs(x45 - exp(t45)));

fprintf('Max error Chebyshev @ nodes : %.3e\n', max_err_cheb);
fprintf('Max error ODE45 (dense out) : %.3e\n', max_err_ode45);

% ====== Plot perbandingan
figure; hold on; grid on; box on;
tt = linspace(t0,tf,400); plot(tt, exp(tt), 'k-', 'LineWidth',1.5);          % eksak
plot(t45, x45, 'b--', 'LineWidth',1.2);                                      % ode45
plot(t_nodes, Xc, 'ro', 'MarkerSize',6, 'LineWidth',1.2);                    % Chebyshev nodes
legend('Solusi eksak e^t','ode45','Chebyshev (CGL nodes)','Location','NorthWest');
xlabel('t'); ylabel('x(t)');
title(sprintf('x''=x, x(0)=1   |   N=%d CGL nodes', N));

% ====== Plot error di node Chebyshev
figure; stem(t_nodes, Xc - x_exact_nodes, 'filled'); grid on; 
xlabel('t (node CGL)'); ylabel('error (Chebyshev - exact)');
title('Galat Chebyshev pada node CGL');

% ================= Helper: CGL nodes & Chebyshev differentiation matrix ================
function [x, D] = chebDiffMat(N)
% Return CGL nodes x (size N+1, descending from 1 to -1) and Chebyshev D
% Trefethen, "Spectral Methods in MATLAB" (CGL formulation)

% Nodes
k  = (0:N)';
x  = cos(pi*k/N);    % x0=1, xN=-1

% c coefficients
c        = ones(N+1,1); 
c(1)     = 2; 
c(end)   = 2;
c        = c .* (-1).^k;

% Build D
X  = repmat(x,1,N+1);
dX = X - X.';              % x_i - x_j
D  = (c*(1./c)')./(dX);    % off-diagonal formula
D(1:(N+1)+1:end) = 0;      % set diag to 0 temporarily
D(1:(N+1)+1:end) = -sum(D,2); % diagonal entries so that each row sums to 0

% Fix endpoints diagonal to closed forms (optional – Trefethen's form already OK)
% (Baris di atas sudah memberi diagonal yang konsisten via row-sum-zero.)
end
