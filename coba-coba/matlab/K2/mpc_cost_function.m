function J = mpc_cost_function(U, x0, reference_pose, Np)
    % U: vektor input (rudder_angle) sepanjang Np
    % x0: keadaan awal
    % reference_pose: [x_ref, y_ref, psi_ref]

    w_pos = 1e-4;
    w_psi = 0.1;
    w_u = 1e-2;

    x = x0;
    J = 0;

    for k = 1:Np
        u = U(k);
        xdot = ship_dynamics(x, u);
        x = x + 1*xdot; % Euler 1 detik

        % Error
        e_pos = norm(x(3:4) - reference_pose(1:2));
        e_psi = angdiff(x(5), reference_pose(3));
        e_psi = abs(e_psi);

        J = J + w_pos*e_pos^2 + w_psi*e_psi^2 + w_u*u^2;
    end
end