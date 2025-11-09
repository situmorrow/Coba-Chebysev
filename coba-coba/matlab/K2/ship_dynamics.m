function xdot = ship_dynamics(x, u)
    % State: [v; r; x_ship; y_ship; psi]
    % Input: rudder_angle

    % Parameter kapal
    Lpp = 101.07;
    os_surge = 15.4;

    % Koefisien model (dari CasADi sebelumnya)
    a11 = -0.0214; a12 = -0.0638;
    a21 = -0.0119; a22 = -0.0556;
    b11 = 0.01; b12 = 0.0099;

    v = x(1); r = x(2); psi = x(5);
    rudder = u;

    % Dinamika
    v_dot = a11*v + a12*r + b11*rudder;
    r_dot = a21*v + a22*r + b12*rudder;
    x_dot = os_surge*cos(psi) - v*sin(psi);
    y_dot = os_surge*sin(psi) + v*cos(psi);
    psi_dot = r;

    xdot = [v_dot; r_dot; x_dot; y_dot; psi_dot];
end