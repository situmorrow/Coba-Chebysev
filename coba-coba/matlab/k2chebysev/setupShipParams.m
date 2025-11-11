function params = setupShipParams()
    % Setup ship parameters and hydrodynamic coefficients
    % Based on NMPC.txt - Corvette SIGMA class
    
    % Ship geometry
    params.Lpp = 101.07;      % Length between perpendiculars (m)
    params.B = 14;            % Breadth (m)
    params.T = 3.7;           % Draught (m)
    params.m = 2423 * 1e3;    % Displacement (kg)
    params.os_surge = 15.4;   % Surge velocity (m/s)
    params.CB = 0.65;         % Block coefficient
    params.xG = 5.25;         % Center of gravity (m)
    params.rho = 1024;        % Sea water density (kg/m^3)
    params.Adelta = 5.7224;   % Rudder area (m^2)
    params.gyration = 0.156 * params.Lpp; % Radius of gyration (m)
    
    % Dimensionless ratios
    B_L = params.B / params.Lpp;
    B_T = params.B / params.T;
    T_L = params.T / params.Lpp;
    pi_val = pi;
    
    % Hydrodynamic coefficients (nonlinear model from NMPC.txt)
    params.Yvdot = -((1 + 0.16*params.CB*B_T - 5.1*B_L^2) * pi_val * T_L^2);
    params.Yrdot = -((0.67*B_L - 0.0033*B_T^2) * pi_val * T_L^2);
    params.Nvdot = -((1.1*B_L - 0.041*B_T) * pi_val * T_L^2);
    params.Nrdot = -(((1/12) + 0.017*params.CB*B_T - 0.33*B_L) * pi_val * T_L^2);
    params.Yv = -((1 + 0.4*params.CB*B_T) * pi_val * T_L^2);
    params.Yr = -((-0.5 + 2.2*B_L - 0.08*B_T) * pi_val * T_L^2);
    params.Nv = -((0.5 + 2.4*T_L) * pi_val * T_L^2);
    params.Nr = -((0.25 + 0.039*B_T - 0.56*B_L) * pi_val * T_L^2);
    
    % Additional parameters
    params.Ydelta = params.rho * pi_val * params.Adelta / (4 * params.Lpp * params.T);
    params.Ndelta = -0.5 * params.Ydelta;
    params.Ir = (params.m * params.gyration^2) / (0.5 * params.rho * params.Lpp^5);
    params.Iz = (params.m * (params.xG^2)) / (0.5 * params.rho * params.Lpp^5) + params.Ir;
    
    % Nondimensional parameters
    params.nd_u = 1;
    params.nd_m = params.m / (0.5 * params.rho * params.Lpp^3);
    params.nd_xG = params.xG / params.Lpp;
    
    % System matrices for nonlinear model
    M = [params.nd_m - params.Yvdot, params.nd_m*params.nd_xG - params.Yrdot;
         params.nd_m*params.nd_xG - params.Nvdot, params.Iz - params.Nrdot];
    N = [-params.Yv, params.nd_m*params.nd_u - params.Yr;
         -params.Nv, params.nd_m*params.nd_xG*params.nd_u - params.Nr];
    
    model_A = -M \ N;
    a11 = model_A(1,1); a12 = model_A(1,2);
    a21 = model_A(2,1); a22 = model_A(2,2);
    
    b = [0.01; 1];
    b11 = b(1); b12 = b(2);
    model_B = [b11; b12/params.Lpp];
    
    % Final model matrices (scaled by velocity)
    os_U = params.os_surge;
    params.os_A = os_U * [a11, a12*params.Lpp; a21/params.Lpp, a22] / params.Lpp;
    params.os_B = os_U^2 * model_B / params.Lpp;
    
    % Constraints
    tactical_diameter = 948;
    params.r_max = 0.0932;
    params.r_min = -params.r_max;
    params.u_max = deg2rad(35);
    params.u_min = -params.u_max;
    params.rrot_max = deg2rad(5);
    params.rrot_min = -params.rrot_max;
    
    % Objective weights
    params.w_position = 1e-4;
    params.w_control = 1e-2;
    params.w_orientation = 0.1;
end