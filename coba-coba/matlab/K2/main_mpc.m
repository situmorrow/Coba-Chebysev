clear; clc; close all;

% Parameter
Np = 60;
sim_time = 300;
dt = 1;
distance_condition = 10;

% Keadaan awal dan referensi
x0 = [0; 0; 0; 0; pi/2];
reference_pose = [500; 2000; 0];

% Batasan
u_min = deg2rad(-35);
u_max = deg2rad(35);
r_min = -0.0932;
r_max = 0.0932;

% Inisialisasi
U0 = zeros(Np,1);
array_state = x0;
control_sequence = [];

distance_to_destination = norm(x0(3:4) - reference_pose(1:2));
mpciter = 0;

while distance_to_destination > distance_condition && mpciter < sim_time/dt
    % Optimasi dengan fmincon
    options = optimoptions('fmincon','Display','off');
    [Uopt, ~] = fmincon(@(U) mpc_cost_function(U, x0, reference_pose, Np), ...
                        U0, [], [], [], [], u_min*ones(Np,1), u_max*ones(Np,1), [], options);

    % Terapkan input pertama
    u_opt = Uopt(1);
    xdot = ship_dynamics(x0, u_opt);
    x0 = x0 + dt*xdot;
    x0(5) = mod(x0(5), 2*pi);

    % Simpan
    array_state = [array_state, x0];
    control_sequence = [control_sequence; u_opt];

    % Update
    distance_to_destination = norm(x0(3:4) - reference_pose(1:2));
    U0 = [Uopt(2:end); Uopt(end)]; % Shift
    mpciter = mpciter + 1;
end

% Plot
figure;
plot(array_state(3,:), array_state(4,:), '-b');
hold on;
plot(reference_pose(1), reference_pose(2), 'rx');
xlabel('X (m)');
ylabel('Y (m)');
title('Trayektori Kapal');
grid on;