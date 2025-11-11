function xdot = shipModel(state, control, params)
    % Nonlinear ship dynamics model
    % state = [v; r; x_ship; y_ship; psi]
    % control = rudder_angle (rad)
    
    % Extract states
    v = state(1);
    r = state(2);
    psi = state(5);
    
    % Extract parameters
    os_surge = params.os_surge;
    os_A = params.os_A;
    os_B = params.os_B;
    
    % Nonlinear dynamics
    xdot_dynamics = os_A * [v; r] + os_B * control;
    
    vdot = xdot_dynamics(1);
    rdot = xdot_dynamics(2);
    
    % Kinematics
    xdot_ship = os_surge * cos(psi) - v * sin(psi);
    ydot_ship = os_surge * sin(psi) + v * cos(psi);
    psidot = r;
    
    xdot = [vdot; rdot; xdot_ship; ydot_ship; psidot];
end