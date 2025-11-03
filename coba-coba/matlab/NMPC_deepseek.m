clear; clc;

%% Parameter Simulasi
dt = 0.1;                 % Sampling time
T = 10;                   % Total waktu simulasi
N = 20;                   % Prediction horizon
sim_step = T/dt;          % Jumlah langkah simulasi

%% Parameter Kapal
L = 1.0;                  % Panjang kapal (untuk model)

%% Batas Input dan State
u_min = -0.5;             % Kecepatan minimal (m/s)
u_max = 1.5;              % Kecepatan maksimal (m/s)
delta_min = -pi/4;        % Sudut kemudi minimal (rad)
delta_max = pi/4;         % Sudut kemudi maksimal (rad)

%% Inisialisasi Variabel
x = zeros(3, sim_step+1); % State [x; y; psi]
u = zeros(2, sim_step);   % Input [u; delta]
x0 = [0; 0; 0];           % State awal
x(:,1) = x0;

%% Referensi Trajektori
ref = zeros(3, sim_step+1);
for k = 1:sim_step+1
    t = (k-1)*dt;
    ref(1,k) = 2*t;       % x_ref = 2t
    ref(2,k) = 2*sin(t);  % y_ref = 2*sin(t)
    ref(3,k) = atan2(2*cos(t), 2); % psi_ref = arctan(y_ref'/x_ref')
end

%% Opsi Optimisasi
options = optimoptions('fmincon', ...
    'Display', 'iter-detailed', ...
    'MaxIterations', 100, ...
    'Algorithm', 'sqp');

%% Simulasi NMPC
for k = 1:sim_step
    % Dapatkan state saat ini
    current_x = x(:,k);
    
    % Solve NMPC
    u0 = repmat([0.5; 0], N, 1); % Tebakan awal input
    [u_opt, fval] = fmincon(@(u) costFunction(u, current_x, ref, k, dt, N), ...
        u0, [], [], [], [], ...
        repmat([u_min; delta_min], N, 1), ...
        repmat([u_max; delta_max], N, 1), ...
        @(u) nonlinearConstraints(u, current_x, dt, N), options);
    
    % Ambil input pertama
    u(:,k) = u_opt(1:2);
    
    % Simulasi sistem dengan input yang dihasilkan
    x(:,k+1) = simulateShip(current_x, u(:,k), dt);
end

%% Plot Hasil
figure;
subplot(2,1,1);
plot(x(1,:), x(2,:), 'b-', ref(1,:), ref(2,:), 'r--');
legend('Actual', 'Reference');
title('Trajektori Posisi'); xlabel('x'); ylabel('y');
grid on;

subplot(2,1,2);
plot(t, u(1,:), t, u(2,:));
legend('Kecepatan (u)', 'Sudut Kemudi (delta)');
title('Input Kontrol'); xlabel('Waktu (s)'); ylabel('Nilai');
grid on;

%% Fungsi Biaya (Objective Function)
function J = costFunction(u, x0, ref, k, dt, N)
    J = 0;
    x = x0;
    for i = 1:N
        % Ekstrak input untuk langkah i
        u_i = u(2*(i-1)+1 : 2*i);
        
        % Update state dengan model diskrit
        x = simulateShip(x, u_i, dt);
        
        % Hitung error terhadap referensi
        error = x - ref(:, min(k+i, size(ref,2)));
        
        % Tambahkan ke biaya (weighted)
        J = J + error' * diag([1; 1; 0.5]) * error;
    end
end

%% Fungsi Simulasi Kapal (Model Diskrit)
function x_next = simulateShip(x, u, dt)
    % Model kinematik kapal
    psi = x(3);
    v = u(1);
    delta = u(2);
    
    % Dynamic equations (model diskrit Euler)
    x_next = x + dt * [...
        v * cos(psi);
        v * sin(psi);
        v * tan(delta) / 1.0]; % 1.0 adalah panjang kapal L
end

%% Fungsi Kendala Nonlinier
function [c, ceq] = nonlinearConstraints(u, x0, dt, N)
    ceq = [];
    c = zeros(N, 1);
    x = x0;
    
    for i = 1:N
        u_i = u(2*(i-1)+1 : 2*i);
        x = simulateShip(x, u_i, dt);
        
        % Contoh kendala: hindari area tertentu (contoh: lingkaran di (3,3))
        c(i) = 1 - ((x(1)-3)^2 + (x(2)-3)^2); % Kendala: harus di luar lingkaran
    end
end