% Chebysev_NMPC_fix.m
% NMPC Chebyshev + anti-orbiting: LOS→final heading blend, terminal shaping,
% smoothing, dan stop berbasis prediksi (min jarak di horizon).
clc; clear; close all;
addpath('D:\TES\casadi-3.7.1-windows64-matlab2018b');   % <- sesuaikan path CasADi
import casadi.*;

%% ===== User params =====
time_sampling = 1.0;     % [s] receding step
T = 60.0;                % [s] panjang horizon kontinu
N = 20;                  % orde Chebyshev (jumlah node = N+1)
Np = N;
situation = 'cheb_collision_free';
axis_lim = [-200 5000 -200 2800];

% batasan
r_max = 0.0932;  r_min = -r_max;               % yaw rate [rad/s]
u_max = deg2rad(35); u_min = -u_max;           % rudder [rad]
rrot_max = deg2rad(7);  rrot_min = -rrot_max;  % rudder rate [rad/s]

% bobot cost
w_pos   = 5;          % posisi (stage)
w_yaw   = 0.02;       % heading error wrt blended-ref (stage)
w_u     = 1e-4;       % |u| (stage)
w_ur    = 2e-3;       % |du/dt| (stage)
w_r     = 1e-2;       % |r| (stage)

w_pos_T = 20*w_pos;   % terminal
w_yaw_T = 5*w_yaw;
w_r_T   = 5e-2;
w_u_T   = 1e-2;

% penalti kecepatan tangensial terminal (opsional, membantu anti-orbit)
w_tan_T = 5;

% near-goal blending (LOS -> psi_ref)
R_switch = 250;       % [m] mulai blend heading ref
sigma_sw = 60;        % [m] kemiringan transisi (kecil = tajam)

% kriteria berhenti
R_arrive = 60;                 % [m] radius sukses (jarak aktual)
r_thresh = 0.01;               % [rad/s] |r| kecil pada node terdekat
u_thresh = deg2rad(1);         % [rad]   |u| kecil pada node terdekat
simulation_time = 300;         % [s] max simulasi

%% ===== States & controls =====
% [v; r; x; y; psi], u = delta
v=SX.sym('v'); r=SX.sym('r'); xs=SX.sym('xs'); ys=SX.sym('ys'); psi=SX.sym('psi');
states = [v; r; xs; ys; psi];   n_states = 5;
delta  = SX.sym('delta');       controls = delta; n_controls = 1;

%% ===== SIGMA-class model (seperti kodingan lama) =====
Lpp=101.07; B=14; Td=3.7; m=2423e3; os_surge=15.4;
CB=0.65; xG=5.25; rho=1024; Adelta=5.7224; gyr=0.156*Lpp;

Yvdot= -((1+0.16*CB*(B/Td)-5.1*(B/Lpp)^2)*pi*(Td/Lpp)^2);
Yrdot= -((0.67*(B/Lpp)-0.0033*(B/Td)^2)*pi*(Td/Lpp)^2);
Nvdot= -((1.1*(B/Lpp)-0.041*(B/Td))*pi*(Td/Lpp)^2);
Nrdot= -(((1/12)+0.017*(CB*B/Td)-0.33*(B/Lpp))*pi*(Td/Lpp)^2);
Yv   = -((1+0.4*(CB*B/Td))*pi*(Td/Lpp)^2);
Yr   = -((-0.5+2.2*(B/Lpp)-0.08*(B/Td))*pi*(Td/Lpp)^2);
Nv   = -((0.5+2.4*(Td/Lpp))*pi*(Td/Lpp)^2);
Nr   = -((0.25+0.039*(B/Td)-0.56*(B/Lpp))*pi*(Td/Lpp)^2);

Ydelta = rho*pi*Adelta/(4*Lpp*Td);
Ndelta = -0.5*Ydelta; %#ok<NASGU>
Ir = (m*(gyr)^2)/(0.5*rho*Lpp^5);
Iz = (m*(xG^2))/(0.5*rho*Lpp^5)+Ir;

nd_u=1; nd_m=m/(0.5*rho*Lpp^3); nd_xG=xG/Lpp;
M=[nd_m-Yvdot nd_m*nd_xG-Yrdot;
   nd_m*nd_xG-Nvdot Iz-Nrdot];
Nmat=[-Yv nd_m*nd_u-Yr;
      -Nv nd_m*nd_xG*nd_u-Nr];
A_lin = -inv(M)*Nmat;

a11=A_lin(1,1); a12=A_lin(1,2); a21=A_lin(2,1); a22=A_lin(2,2);
b = [0.01; 1]; model_B = [b(1); b(2)/Lpp];

os_U=os_surge;
os_A=os_U*[a11 a12*Lpp; a21/Lpp a22]/Lpp;
os_B=os_U^2*model_B/Lpp;

x_dyn = [v; r];
xdot_dyn = os_A*x_dyn + os_B*delta;

xdot = [ xdot_dyn(1);
         xdot_dyn(2);
         os_surge*cos(psi) - x_dyn(1)*sin(psi);
         os_surge*sin(psi) + x_dyn(1)*cos(psi);
         x_dyn(2) ];

f_ode = Function('f_ode',{states,controls},{xdot});

%% ===== Chebyshev nodes/weights/D =====
[taus,w_cc,D0] = cheb_nodes_weights_D(N);    % nodes [-1,1]
taus = flipud(taus);  w_cc = flipud(w_cc);   % urut kecil→besar
D = flipud(fliplr(D0));
alpha = 2/T;                                    % d/dt = (2/T) d/dtau

%% ===== NLP vars =====
S = SX.sym('S', n_states, N+1);   % state di node
U = SX.sym('U', n_controls, N+1); % kontrol di node

% parameter: x0(5); x_ref; y_ref; psi_ref
P = SX.sym('P', n_states + 3);
x0 = P(1:n_states);  xref = P(n_states+1);
yref = P(n_states+2); psiref = P(n_states+3);

%% ===== Utils =====
angwrap = @(th) atan2(sin(th),cos(th));
blend_factor = @(dx,dy,Rsw,sg) 0.5*(1 + tanh( (sqrt(dx.^2+dy.^2)-Rsw)/sg ));

%% ===== Cost =====
J = 0;
for k=1:(N+1)
    sk = S(:,k); uk = U(:,k);
    xk=sk(3); yk=sk(4); psik=sk(5);
    dx = xref - xk; dy = yref - yk;

    % LOS→psi_ref blend
    s_bl = blend_factor(dx,dy,R_switch,sigma_sw);
    chi_k = atan2(dy,dx);
    h_ref = s_bl*chi_k + (1-s_bl)*psiref;
    psi_err = angwrap(psik - h_ref);

    pos_err = [xk - xref; yk - yref];

    J = J ...
        + w_pos*(pos_err.'*pos_err) ...
        + w_yaw*(psi_err.'*psi_err) ...
        + w_u*(uk.'*uk) ...
        + w_r*(sk(2)^2);
end

% penalti rate kontrol via turunan Chebyshev
for k=1:(N+1)
    ur_k = 0;
    for i=1:(N+1), ur_k = ur_k + D(k,i)*U(:,i); end
    ur_k = alpha*ur_k;
    J = J + w_ur*(ur_k.'*ur_k);
end

% terminal cost kuat
sN = S(:,end); uN = U(:,end);
dxN = xref - sN(3); dyN = yref - sN(4);
s_blN = blend_factor(dxN,dyN,R_switch,sigma_sw);
chi_N = atan2(dyN,dxN);
h_refN = s_blN*chi_N + (1-s_blN)*psiref;
psi_errN = angwrap(sN(5) - h_refN);
pos_errN = [sN(3)-xref; sN(4)-yref];

% penalti kecepatan tangensial terminal (anti-orbit)
vtanN = os_surge * sin( angwrap(sN(5) - chi_N) );

J = J ...
   + w_pos_T*(pos_errN.'*pos_errN) ...
   + w_yaw_T*(psi_errN.'*psi_errN) ...
   + w_r_T*(sN(2)^2) ...
   + w_u_T*(uN.'*uN) ...
   + w_tan_T*(vtanN^2);

%% ===== Constraints =====
% kolokasi: alpha * sum_i D(k,i) S_i = f(S_k, U_k)
geq = {};
for k=1:(N+1)
    dS_k = 0;
    for i=1:(N+1), dS_k = dS_k + D(k,i)*S(:,i); end
    geq{end+1} = alpha*dS_k - f_ode(S(:,k), U(:,k)); %#ok<AGROW>
end
% initial condition
geq{end+1} = S(:,1) - x0;

g_fun = vertcat(geq{:});
lbg = zeros(size(g_fun));  ubg = zeros(size(g_fun));

% batas rate kontrol: alpha*D*U ∈ [rrot_min, rrot_max]
g_rate = SX([]); lbg_rate=[]; ubg_rate=[];
for k=1:(N+1)
    ur_k = 0;
    for i=1:(N+1), ur_k = ur_k + D(k,i)*U(:,i); end
    ur_k = alpha*ur_k;
    g_rate = [g_rate; ur_k]; %#ok<AGROW>
    lbg_rate = [lbg_rate; rrot_min];
    ubg_rate = [ubg_rate; rrot_max];
end

% satukan semua kendala
g_all   = [g_fun; g_rate];
lbg_all = [lbg;   lbg_rate];
ubg_all = [ubg;   ubg_rate];

% vektor keputusan z = [reshape(S); reshape(U)]
z  = [reshape(S, n_states*(N+1),1);
      reshape(U, n_controls*(N+1),1)];

% box bounds untuk r (baris-2 S) dan u (blok U)
lbx = -inf*ones(size(z));  ubx =  inf*ones(size(z));
for k=1:(N+1)
    idx_r = sub2ind([n_states, N+1], 2, k);
    lbx(idx_r) = r_min;  ubx(idx_r) = r_max;
end
offsetU = n_states*(N+1);
for k=1:(N+1)
    idx_u = offsetU + k;
    lbx(idx_u) = u_min;  ubx(idx_u) = u_max;
end

%% ===== Solver =====
nlp = struct('f',J,'x',z,'g',g_all,'p',P);
opts = struct;
opts.ipopt.max_iter=800; opts.ipopt.print_level=0;
opts.print_time=0;
opts.ipopt.acceptable_tol=1e-8;
opts.ipopt.acceptable_obj_change_tol=1e-10;
solver = nlpsol('solver','ipopt',nlp,opts);

%% ===== Simulation (receding horizon) =====
x0_val = [0;0;0;0;pi/2];
reference_pose = [4000;2000; 0];   % [xr, yr, psi_ref]
array_state = x0_val;
array_state_history = [];
control_sequence = [];

z0 = zeros(size(z));
mpciter = 0;

while mpciter < simulation_time/time_sampling
    p_val = [x0_val; reference_pose(:)];

    sol = solver('x0',z0,'lbx',lbx,'ubx',ubx,'lbg',lbg_all,'ubg',ubg_all,'p',p_val);
    z_opt = full(sol.x);     % solusi
    z0 = z_opt;              % warm-start

    S_opt = reshape(z_opt(1:n_states*(N+1)), n_states, N+1);
    U_opt = reshape(z_opt(n_states*(N+1)+1:end), n_controls, N+1);

    % simpan horizon untuk gambar (format (Np+2) x 5)
    pred_states = [S_opt.'; S_opt(:,end).'];
    array_state_history(:,:,end+1) = pred_states; %#ok<AGROW>

    % kontrol diterapkan: node pertama
    u_apply = U_opt(:,1);
    control_sequence = [control_sequence; u_apply]; %#ok<AGROW>

    % propagate 1 step
    xdot_now = full(f_ode(x0_val, u_apply));
    next_state = x0_val + time_sampling*xdot_now;
    next_state(5) = mod(next_state(5), 2*pi);
    x0_val = next_state;
    array_state(:,end+1) = x0_val; %#ok<AGROW>

    mpciter = mpciter + 1;

    %% === STOP CONDITION: pakai jarak minimum di horizon prediksi ===
    dist_nodes = sqrt( (S_opt(3,:) - reference_pose(1)).^2 + ...
                       (S_opt(4,:) - reference_pose(2)).^2 );
    [dist_min, idx_min] = min(dist_nodes);
    r_at_min = S_opt(2, idx_min);
    u_at_min = U_opt(1, idx_min);

    dist_actual = norm(x0_val(3:4) - reference_pose(1:2));

    if ( (dist_min <= R_arrive && abs(r_at_min)<=r_thresh && abs(u_at_min)<=u_thresh) ...
         || (dist_actual <= R_arrive) )
        fprintf('MISSION COMPLETE at iter %d (dist_min=%.2f m)\n', mpciter, dist_min);
        break;
    end
end

%% ===== Output untuk draw =====
time = zeros(1, mpciter + Np + 2);
for i=2:numel(time), time(i)=time(i-1)+time_sampling; end
total_time_index = numel(time);

if isempty(array_state_history)
    last_hist = repmat(array_state(:,end).', Np+2, 1);
else
    last_hist = array_state_history(:,:,end);
end
state_last = last_hist(2:end,:).';
array_state = cat(2, array_state, state_last);

% sanity
assert(size(array_state,1)==5);
assert(size(array_state,2)==total_time_index);
assert(size(array_state_history,1)==Np+2 && size(array_state_history,2)==5);

% gunakan fungsi gambar lamamu:
draw_collision_free(array_state, array_state_history, ...
    reference_pose(:), total_time_index, axis_lim, Np, situation, time);

%% ===== Helpers =====
function [taus,w,D] = cheb_nodes_weights_D(N)
    k = (0:N)'; 
    taus = cos(pi*k/N);   % nodes [-1,1]

    % Clenshaw–Curtis weights (aproksimasi stabil)
    w = zeros(N+1,1);
    if N==0
        w = 2;
    else
        for j=0:N
            s = 0;
            for m=1:floor(N/2)
                s = s + (2/(1-4*m^2))*cos(2*m*pi*j/N);
            end
            w(j+1) = (2/N)*(1 - s);
        end
    end

    % Differentiation matrix (Trefethen)
    c = [2; ones(N-1,1); 2].*(-1).^k;
    X = repmat(taus,1,N+1);
    dX = X - X.';
    D = (c*(1./c)')./(dX + eye(N+1));
    D = D - diag(sum(D,2));
end
