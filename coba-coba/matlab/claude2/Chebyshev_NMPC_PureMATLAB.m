function Chebyshev_NMPC_PureMATLAB
clc; clear; close all;

%% ======================= HORIZON & DISKRETISASI PSM =====================
Ts   = 1.0;              % sampling eksekusi MPC (s)
N    = 20;               % orde polinomial Chebyshev (jumlah node = N+1)
T    = 60;               % panjang horizon kontinu (detik) -> dipetakan ke tau∈[-1,1]
                          % skala turunan: ds/dtau = (T/2) f(s,u)  [PDF 1.4.2]
% Node CGL, D-matrix, weights
[tau, D] = cheb_nodes_and_D(N);                 % node τ_i = cos(iπ/N), D (CGL) 
w        = cgl_weights(N);                      % bobot kuadratur w_k (genap/ganjil)
scale_dtau = 2/T;                                % 2/T, dipakai pada kendala kolokasi

%% ======================= MODEL KAPAL & KONSTAN ==========================
% ---- parameter fisik dari skrip NMPC CasADi milikmu ----
Lpp = 101.07;  B = 14;  Tdraft = 3.7; m = 2423e3;
os_surge = 15.4;  CB = 0.65;  xG = 5.25;  rho = 1024;
Adelta = 5.7224; gyration = 0.156*Lpp;

% Koefisien hidrodinamika & model nondim (mengacu skripmu)
[Yvdot, Yrdot, Nvdot, Nrdot, Yv, Yr, Nv, Nr, ...
 Ydelta, Ndelta, Ir, Iz, nd_u, nd_m, nd_xG, os_A, os_B] = ship_coefficients(...
    Lpp,B,Tdraft,m,CB,xG,rho,Adelta,gyration,os_surge);              % :contentReference[oaicite:8]{index=8}

% Ukuran state & kontrol
nx = 5;  % [v; r; x; y; psi]
nu = 1;  % rudder angle

% Batasan dari skripmu
r_max = 0.0932; r_min = -r_max;
u_max = deg2rad(35); u_min = -u_max;
rrot_max = deg2rad(5); rrot_min = -rrot_max;

% Matriks output untuk tracking (x,y,psi)
C = [0 0 1 0 0;
     0 0 0 1 0;
     0 0 0 0 1];

% Bobot objektif
Qpos  = diag([1e-4, 1e-4, 0.1]);    % (x,y,psi) — konsisten semangat skrip
R     = 1e-2;                        % effort kontrol

% Target
xref = [500; 2000; 0];               % (x,y,psi) target
x0   = [0;0;0;0;pi/2];               % kondisi awal

% MPC eksekusi
sim_time = 300;      % detik eksekusi
Kmax     = floor(sim_time/Ts);

% Warm start
S0 = repmat(x0(:).', N+1, 1);              % tebakan awal state di node
U0 = zeros(N+1,1);                         % tebakan awal kontrol di node

% Precompute indeks pembungkus variabel keputusan
nzS  = (N+1)*nx;  nzU = (N+1)*nu;  
z0   = [S0(:); U0(:)];

% Batas kotak variabel
lbS = -inf(nzS,1); ubS =  inf(nzS,1);
% batasi yaw rate r (komponen ke-2 state)
r_idx = (1:(N+1))'*nx - (nx-2);  % posisi r dalam vektor [v r x y psi] berturut
lbS(r_idx) = r_min;  ubS(r_idx) = r_max;

lbU = u_min*ones(nzU,1);  ubU = u_max*ones(nzU,1);
lbz = [lbS; lbU];  ubz = [ubS; ubU];

% Opsi fmincon
opts = optimoptions('fmincon','Algorithm','sqp',...
    'MaxIterations', 400, 'Display','iter', ...
    'ConstraintTolerance',1e-6,'OptimalityTolerance',1e-6,'StepTolerance',1e-8);

% Jalankan MPC
x_exec = x0;  u_applied_hist = [];  x_hist = x0(:).';
for kexec = 1:Kmax
    % fungsi objektif & kendala untuk kondisi awal x_exec
    objfun  = @(z) objective_cheb(z,N,nx,nu,C,Qpos,R,xref,w,T);
    nonlcon = @(z) constraints_cheb(z,N,nx,nu,x_exec,D,scale_dtau,...
                                    rrot_min,rrot_max,T,os_A,os_B,os_surge);
    % solve
    [zsol,~,exitflag] = fmincon(objfun, z0, [],[],[],[], lbz, ubz, nonlcon, opts);
    if exitflag<=0
        warning('FMINCON tidak konvergen, gunakan solusi terakhir.');
        zsol = z0;
    end

    % Ambil kontrol node pertama sebagai kontrol eksekusi (receding horizon)
    [Ssol, Usol] = unpack_SU(zsol, N, nx, nu);
    u_apply = Usol(1);  % u(τ_0)
    u_applied_hist(end+1,1) = u_apply;

    % Simulasi 1 langkah nyata (Euler kecil) dengan Ts
    x_exec = x_exec + Ts * ship_f(x_exec, u_apply, os_A, os_B, os_surge);
    x_exec(5) = wrapToPi(x_exec(5));
    x_hist(end+1,:) = x_exec(:).';

    % Warm-start geser (shift) untuk iterasi berikutnya
    z0 = shift_warm_start(Ssol,Usol,nx,nu);
    
    % Berhenti jika dekat target
    if norm(x_exec(3:4)-xref(1:2)) < 10
        break;
    end
end

% Plot ringkas
tvec = 0:Ts:(size(x_hist,1)-1)*Ts;
figure; plot(tvec, x_hist(:,3), tvec, x_hist(:,4)); grid on
xlabel('t [s]'); ylabel('posisi (m)'); legend('x','y');
figure; plot(tvec, rad2deg(wrapToPi(x_hist(:,5))), '--'); hold on
yline(rad2deg(xref(3)),'r--'); grid on; xlabel('t [s]'); ylabel('\psi [deg]');
figure; stairs(0:Ts:Ts*(numel(u_applied_hist)-1), rad2deg(u_applied_hist)); grid on
xlabel('t [s]'); ylabel('rudder [deg]');

end

%% ========================= OBJEKTIF CHEBYSHEV ===========================
function J = objective_cheb(z,N,nx,nu,C,Qpos,R,xref,w,T)
    [S,U] = unpack_SU(z,N,nx,nu);
    % h = C*s  (x,y,psi)
    H = (C*S.').';                % (N+1) x 3
    E = H - repmat(xref(:).',N+1,1);
    % bungkus error heading ke [-pi,pi]
    E(:,3) = wrapToPi(E(:,3));
    % Kuadratur CGL: J ≈ (T/2) * sum_k w_k * ( e'Qe + u'Ru )
    Jtrack = sum( w(:) .* rowquad(E,Qpos) );
    Jeff   = sum( w(:) .* (U.^2) * R );
    phiCOLREGS = 0; % placeholder hukuman COLREGS
    J = (T/2)*(Jtrack + Jeff + phiCOLREGS);
end

function val = rowquad(E,Q)
    % hitung e'Qe per baris
    val = sum((E*Q).*E,2);
end

%% ======================== KENDALA CHEBYSHEV =============================
function [c,ceq] = constraints_cheb(z,N,nx,nu,x0,D,scale_dtau,...
                                    rdot_min,rdot_max,T,os_A,os_B,os_surge)
    [S,U] = unpack_SU(z,N,nx,nu);     % S: (N+1)x5, U: (N+1)x1
    % --- 1) Kolokasi Chebyshev: (2/T) * sum_j D_kj s_j = f(s_k,u_k)
    % Bentuk vektor semua node:
    fku = zeros(N+1,nx);
    for k=1:N+1
        fku(k,:) = ship_f(S(k,:).', U(k), os_A, os_B, os_surge).';
    end
    ceq_dyn = scale_dtau*(D*S) - fku;             % (N+1) x nx
    ceq_dyn = ceq_dyn(:);

    % --- 2) Kondisi awal di node τ_0 (k=1)
    ceq_ic = S(1,:).' - x0(:);

    % --- 3) Batas laju kontrol: u̇ = (2/T)*D*U \in [rdot_min, rdot_max]
    %    (memakai notasi umum u_dot_min/max; di sini pakai rrot_min/max)
    u_dot = scale_dtau*(D*U);      % (N+1)x1
    c_u_dot = [ u_dot - rdot_max;  % <=0
               -u_dot + rdot_min];

    % Gabungkan
    ceq = [ceq_dyn; ceq_ic];
    c   = c_u_dot;
end

%% =========================== UTIL ITAS ==================================
function [S,U] = unpack_SU(z,N,nx,nu)
    nzS = (N+1)*nx;
    S = reshape(z(1:nzS), [N+1, nx]);
    U = reshape(z(nzS+1:end), [N+1, nu]);
end

function z_shift = shift_warm_start(S,U,nx,nu)
    % shift 1 langkah node ke depan, duplikasi node terakhir
    Snext = [S(2:end,:); S(end,:)];
    Unext = [U(2:end,:); U(end,:)];
    z_shift = [Snext(:); Unext(:)];
end

function xdot = ship_f(x,u,os_A,os_B,os_surge)
    % x = [v; r; X; Y; psi]
    v   = x(1);  r = x(2);  psi = x(5);
    xdot_lin = os_A*[v;r] + os_B*u; % (v_dot, r_dot)
    vdot = xdot_lin(1);
    rdot = xdot_lin(2);
    Xdot =  os_surge*cos(psi) - v*sin(psi);
    Ydot =  os_surge*sin(psi) + v*cos(psi);
    psidot = r;
    xdot = [vdot; rdot; Xdot; Ydot; psidot];
end

%% ======= Node Chebyshev–Gauss–Lobatto & Matriks Diferensiasi D =========
function [tau, D] = cheb_nodes_and_D(N)
    % Node CGL
    k   = (0:N).';
    tau = cos(pi*k/N);                                % τ_k = cos(kπ/N)  :contentReference[oaicite:9]{index=9}
    % D-matrix (Trefethen/standard) sesuai PDF:
    c = [2; ones(N-1,1); 2].* (-1).^k;               % c_i = 2 untuk i=0,N; 1 lainnya
    X = repmat(tau,1,N+1);
    dX = X - X.';
    D = (c*(1./c)')./(dX + eye(N+1));
    D = D - diag(sum(D,2));
    % Ubah diagonal tepat sesuai rumus PDF (opsional; hasil di atas sudah konsisten),
    % tetapi kita biarkan bentuk stabil (identik secara numerik).
end

%% ===================== Bobot Kuadratur CGL (γ-formula) ==================
function w = cgl_weights(N)
    % Implementasi sesuai rumus di PDF 1.4.7 untuk N genap & ganjil:
    % - even N: w0=wN=1/(N^2-1), wk = (2/N)[ γ(0,k) + γ(N/2,k) + 2 sum_{i=1}^{N/2-1} γ(i,k) ]
    % - odd  N: w0=wN=1/N^2,     wk = (2/N)[ γ(0,k) + 2 sum_{i=1}^{(N-1)/2} γ(i,k) ]
    % dengan γ(i,k) = (1/(1-4 i^2)) * cos(2π i k / N)
    % (Perhatikan γ(0,k) = 1)
    w = zeros(N+1,1);
    if mod(N,2)==0   % N genap
        w(1)   = 1/(N^2 - 1);
        w(end) = w(1);
        for k = 1:N-1
            s = 1; % γ(0,k)
            s = s + (1/(1-4*(N/2)^2))*cos(2*pi*(N/2)*k/N); % γ(N/2,k)
            for i=1:(N/2-1)
                s = s + 2*(1/(1-4*i^2))*cos(2*pi*i*k/N);
            end
            w(k+1) = (2/N)*s;
        end
    else            % N ganjil
        w(1)   = 1/(N^2);
        w(end) = w(1);
        for k = 1:N-1
            s = 1; % γ(0,k)
            for i=1:(N-1)/2
                s = s + 2*(1/(1-4*i^2))*cos(2*pi*i*k/N);
            end
            w(k+1) = (2/N)*s;
        end
    end
    % Normalisasi kecil agar sum w ≈ 2 (panjang interval [-1,1])
    w = (2/sum(w)) * w;
end

%% ======================== Koefisien Kapal SIGMA =========================
function [Yvdot, Yrdot, Nvdot, Nrdot, Yv, Yr, Nv, Nr, ...
          Ydelta, Ndelta, Ir, Iz, nd_u, nd_m, nd_xG, os_A, os_B] = ...
          ship_coefficients(Lpp,B,Tdraft,m,CB,xG,rho,Adelta,gyration,os_surge)

    Yvdot = -((1+0.16*CB*(B/Tdraft)-5.1*(B/Lpp)^2)*pi*(Tdraft/Lpp)^2);
    Yrdot = -((0.67*(B/Lpp)-0.0033*(B/Tdraft)^2)*pi*(Tdraft/Lpp)^2);
    Nvdot = -((1.1*(B/Lpp)-0.041*(B/Tdraft))*pi*(Tdraft/Lpp)^2);
    Nrdot = -(((1/12)+0.017*(CB*B/Tdraft)-0.33*(B/Lpp))*pi*(Tdraft/Lpp)^2);
    Yv = -((1+0.4*(CB*B/Tdraft))*pi*(Tdraft/Lpp)^2);
    Yr = -((-0.5+2.2*(B/Lpp)-0.08*(B/Tdraft))*pi*(Tdraft/Lpp)^2);
    Nv = -((0.5+2.4*(Tdraft/Lpp))*pi*(Tdraft/Lpp)^2);
    Nr = -((0.25+0.039*(B/Tdraft)-0.56*(B/Lpp))*pi*(Tdraft/Lpp)^2);

    Ydelta = rho*pi*Adelta/(4*Lpp*Tdraft);
    Ndelta = -0.5*Ydelta;
    Ir = (m*gyration^2)/(0.5*rho*Lpp^5);
    Iz = (m*(xG^2))/(0.5*rho*Lpp^5)+Ir;

    nd_u = 1;
    nd_m = m/(0.5*rho*Lpp^3);
    nd_xG = xG/Lpp;

    M = [nd_m-Yvdot,          nd_m*nd_xG - Yrdot;
         nd_m*nd_xG - Nvdot,  Iz - Nrdot];
    Nmat = [-Yv,             nd_m*nd_u - Yr;
            -Nv,             nd_m*nd_xG*nd_u - Nr];

    model_A = - (M\Nmat);
    a11 = model_A(1,1); a12 = model_A(1,2);
    a21 = model_A(2,1); a22 = model_A(2,2);

    b = [0.01; 1];
    b11 = b(1); b12 = b(2);
    model_B = [b11; b12/Lpp];

    os_U = os_surge;
    os_A = os_U*[a11 a12*Lpp; a21/Lpp a22]/Lpp;
    os_B = (os_U^2)*model_B/Lpp;
end
