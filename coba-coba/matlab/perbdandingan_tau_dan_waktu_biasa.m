clc; clear; close all;

% Parameter waktu
t0 = 0; tf = 10;
t = linspace(t0, tf, 200);
tau = (2*t - (tf + t0)) / (tf - t0);

% Fungsi dalam domain t
f_t = t.^2;

% Fungsi dalam domain tau
f_tau = 25*(tau + 1).^2;

% Plot perbandingan
figure;
subplot(2,1,1);
plot(t, f_t, 'b', 'LineWidth', 2);
xlabel('t (detik)'); ylabel('f(t)');
title('Fungsi dalam domain waktu biasa: f(t) = t^2');
grid on;

subplot(2,1,2);
plot(tau, f_tau, 'r', 'LineWidth', 2);
xlabel('\tau (non-dimensional)'); ylabel('f(\tau)');
title('Fungsi dalam domain non-dimensional: f(\tau) = 25(\tau + 1)^2');
grid on;
