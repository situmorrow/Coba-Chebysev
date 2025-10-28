function nmpc_ship_demo()
% ================================================================
% NMPC kapal sederhana (pure MATLAB, tanpa toolbox)
% Cost: sum_i ||[x;y;psi_des]-[x;y;psi]||_Q^2 + Rdu*||Δu||^2 + Ru*u^2 + Qr*r^2
% - Heading pakai LOS (psi_des = atan2(y_ref - y, x_ref - x)) => anti orbit.
% - u dibatasi [u_min,u_max] via tanh (z->u).
% - Dinamika: xdot = v*cos(psi), ydot = v*sin(psi), psidot = r, rdot = -a r + b u.
% ================================================================

clc; clear; close all;

% ---------- Parameter model ---------- done
a = 0.5;     % redaman yaw
b = 1.0;     % gain rudder
v = 1.0;     % kecepatan konstan

% ---------- Waktu & horizon ---------- done
dt  = 0.1;
Np  = 30;    % horizon lebih panjang agar look-ahead baik
Tsim = 20;

% ---------- Bobot biaya ---------- done
Q   = diag([12, 12, 6]);  % tracking [x y psi_des]
Rdu = 0.30;               % penalti Δu
Ru  = 0.05;               % penalti usaha u
Qr  = 0.80;               % penalti yaw-rate r
P   = diag([60, 60, 30]); % terminal weight besar

% ---------- Batas input ---------- done
u_min = -0.8;  u_max = 0.8;
du_min = -0.08; du_max = 0.08;

% ---------- Inisialisasi ---------- done
s = [0; 0; 0; 0];        % [x; y; psi; r]
u_prev = 0;
s_hist = s;  u_hist = [];
t = 0:dt:Tsim;

% ---------- Referensi lintasan ---------- done
% target (10,0), heading absolut tidak dipakai—LOS yang dipakai
xref = @(tk) [min(10, 0.5*tk); 0; 0];

% ---------- Optimizer bawaan (Nelder–Mead) ----------
z_opt = zeros(Np,1);  % keputusan di ruang tak-terbatas
opts = optimset('Display','off','MaxFunEvals',2e4,'TolX',1e-5,'TolFun',1e-5);

fprintf('Simulasi NMPC kapal dengan LOS...\n');
for k = 1:length(t)-1
    costfun = @(z) cost_ship_LOS( ...
        z, s, u_prev, a,b,v,dt,Np, Q,Rdu,Ru,Qr,P, xref, ...
        u_min,u_max, du_min,du_max);

    [z_opt, ~] = fminsearch(costfun, z_opt, opts);

    u_seq = z2u(z_opt, u_min, u_max);
    u = u_seq(1);
    u_hist(end+1) = u;

    s = rk4(@(ss,uu) dyn_ship(ss,uu,a,b,v), s, u, dt);
    s_hist(:,end+1) = s;

    u_prev = u;
    z_opt = [z_opt(2:end); z_opt(end)];  % warm start
end

% ===================== Plot utama =========================
figure('Color','w');
subplot(3,1,1); hold on; grid on; axis equal;
plot(s_hist(1,:), s_hist(2,:), 'LineWidth',1.6, 'Color',[0.95 0.6 0.1]);
plot(0,0,'go','MarkerFaceColor','g');     % start
plot(10,0,'r*','MarkerSize',10);          % target
xlabel('x [m]'); ylabel('y [m]'); title('Lintasan kapal (LOS-NMPC)');

subplot(3,1,2); hold on; grid on;
plot(t, s_hist(3,:), 'LineWidth',1.6);
ylabel('\psi [rad]'); title('Heading');

subplot(3,1,3); hold on; grid on;
stairs(t(1:end-1), u_hist, 'LineWidth',1.6);
yline(u_min,'--r'); yline(u_max,'--r');
ylabel('u [rad]'); xlabel('waktu [s]');
title('Input rudder');

% ===================== Animasi (trail + marker) ===========
figure('Color','w'); hold on; grid on; axis equal;
title('Animasi lintasan kapal (NMPC)');
xlabel('x [m]'); ylabel('y [m]');
plot(10,0,'r*','MarkerSize',10);
plot(0,0,'go','MarkerFaceColor','g');
xlim([min(s_hist(1,:))-1, max([s_hist(1,:),10])+1]);
ylim([min(s_hist(2,:))-1, max([s_hist(2,:),0])+1]);

trail = plot(NaN,NaN,'-','LineWidth',1.6, 'Color',[0.95 0.6 0.1]);
ship  = plot(NaN,NaN,'bo','MarkerFaceColor','b');
head  = quiver(0,0,0,0,0.6,'MaxHeadSize',2,'AutoScale','off');

for k = 1:length(s_hist)
    set(trail,'XData',s_hist(1,1:k),'YData',s_hist(2,1:k));
    set(ship,'XData',s_hist(1,k),'YData',s_hist(2,k));
    psi = s_hist(3,k);
    set(head,'XData',s_hist(1,k),'YData',s_hist(2,k), ...
             'UData',cos(psi),'VData',sin(psi));
    drawnow; pause(0.02);
end
end

% ===================== COST (LOS) ==========================
function J = cost_ship_LOS(z, s0, u_prev, a,b,v,dt,Np, Q,Rdu,Ru,Qr,P, xref, ...
                           u_min,u_max, du_min,du_max)
u_seq = z2u(z, u_min, u_max);
s = s0;  J = 0;

for i = 1:Np
    % referensi posisi
    href = xref(i*dt);  xr = href(1); yr = href(2);

    % heading desired (LOS) & error sudut dibungkus
    psi_des = atan2(yr - s(2), xr - s(1));
    psi_err = wrapToPi_local(psi_des - s(3));

    % error tracking [x;y;psi_des]
    e = [xr - s(1); yr - s(2); psi_err];

    % perubahan input
    if i==1, du = u_seq(i) - u_prev; else, du = u_seq(i) - u_seq(i-1); end

    % soft constraint Δu
    du_pen = 0;
    if du < du_min, du_pen = du_pen + 1e3*(du_min - du)^2; end
    if du > du_max, du_pen = du_pen + 1e3*(du - du_max)^2; end

    % stage cost
    J = J ...
        + e'*Q*e ...
        + Rdu*(du^2) ...
        + Ru*(u_seq(i)^2) ...
        + Qr*(s(4)^2) ...
        + du_pen;

    % propagate
    s = rk4(@(ss,uu) dyn_ship(ss,uu,a,b,v), s, u_seq(i), dt);
end

% terminal cost (pakai LOS ke posisi akhir horizon)
hrefT = xref(Np*dt);  xrT=hrefT(1); yrT=hrefT(2);
psi_des_T = atan2(yrT - s(2), xrT - s(1));
eT = [xrT - s(1); yrT - s(2); wrapToPi_local(psi_des_T - s(3))];
J = J + eT'*P*eT;
end

% ===================== Dinamika kapal ====================== done
function sdot = dyn_ship(s, u, a,b,v)
% s=[x;y;psi;r]
psi = s(3); r = s(4);
sdot = [ v*cos(psi);
         v*sin(psi);
         r;
        -a*r + b*u ];
end

% ===================== RK4 integrator ======================
function s_next = rk4(f, s, u, h)
k1 = f(s, u);
k2 = f(s + 0.5*h*k1, u);
k3 = f(s + 0.5*h*k2, u);
k4 = f(s + h*k3, u);
s_next = s + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end

% ===================== z -> u (hard bound) =================
function u = z2u(z, u_min, u_max)
u = u_min + (u_max - u_min)*(tanh(z) + 1)/2;
end

% ===================== wrap angle ==========================
function ang = wrapToPi_local(ang)
ang = mod(ang + pi, 2*pi) - pi;
end
