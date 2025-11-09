function xdot = ship_dynamics(x, u, ship_params)
    % State: [v; r; x_ship; y_ship; psi]
    % Input: rudder_angle
    % ship_params: struktur dengan field yang diperlukan
    
    % Extract states
    v = x(1); r = x(2); psi = x(5);
    rudder = u(1);
    
    % Extract parameters
    os_surge = ship_params.os_surge;
    Lpp = ship_params.Lpp;
    
    % Koefisien model (dari perhitungan CASADI sebelumnya)
    a11 = ship_params.a11; a12 = ship_params.a12;
    a21 = ship_params.a21; a22 = ship_params.a22;
    b11 = ship_params.b11; b12 = ship_params.b12;
    
    % Dinamika linear sway-yaw
    v_dot = a11*v + a12*r + b11*rudder;
    r_dot = a21*v + a22*r + b12*rudder;
    
    % Dinamika posisi dan heading
    x_dot = os_surge*cos(psi) - v*sin(psi);
    y_dot = os_surge*sin(psi) + v*cos(psi);
    psi_dot = r;
    
    xdot = [v_dot; r_dot; x_dot; y_dot; psi_dot];
end