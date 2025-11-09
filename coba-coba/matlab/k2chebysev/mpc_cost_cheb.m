function J = mpc_cost_cheb(z, P, N, T, ship_params, weights)
    % Reshape variabel keputusan
    n_states = 5; n_controls = 1; n_nodes = N+1;
    S = reshape(z(1:n_states*n_nodes), n_states, n_nodes);
    U = reshape(z(n_states*n_nodes+1:end), n_controls, n_nodes);
    
    % Extract parameters
    x0 = P(1:n_states);
    xref = P(n_states+1); yref = P(n_states+2); psiref = P(n_states+3);
    os_surge = ship_params.os_surge;
    
    % Extract weights
    w_pos = weights.w_pos; w_yaw = weights.w_yaw; w_u = weights.w_u;
    w_ur = weights.w_ur; w_r = weights.w_r;
    w_pos_T = weights.w_pos_T; w_yaw_T = weights.w_yaw_T;
    w_r_T = weights.w_r_T; w_u_T = weights.w_u_T;
    w_tan_T = weights.w_tan_T;
    R_switch = weights.R_switch; sigma_sw = weights.sigma_sw;
    
    % Precompute
    [taus, w_cc, D] = cheb_nodes_weights_D(N);
    alpha = 2/T;
    
    % Blend factor
    blend_factor = @(dx, dy, Rsw, sg) 0.5 * (1 + tanh((sqrt(dx.^2+dy.^2)-Rsw)/sg));
    
    J = 0;
    
    % Stage cost
    for k = 1:n_nodes
        xk = S(3,k); yk = S(4,k); psik = S(5,k);
        uk = U(1,k);
        
        dx = xref - xk; dy = yref - yk;
        s_bl = blend_factor(dx, dy, R_switch, sigma_sw);
        chi_k = atan2(dy, dx);
        h_ref = s_bl * chi_k + (1 - s_bl) * psiref;
        psi_err = angdiff_custom(psik, h_ref);
        
        pos_err = [xk - xref; yk - yref];
        
        % Control rate
        ur_k = alpha * D(k,1:n_nodes) * U';
        
        J = J + w_pos * (pos_err' * pos_err) + ...
                w_yaw * (psi_err^2) + ...
                w_u * (uk^2) + ...
                w_r * (S(2,k)^2) + ...
                w_ur * (ur_k^2);
    end
    
    % Terminal cost
    sN = S(:,end); uN = U(:,end);
    dxN = xref - sN(3); dyN = yref - sN(4);
    s_blN = blend_factor(dxN, dyN, R_switch, sigma_sw);
    chi_N = atan2(dyN, dxN);
    h_refN = s_blN * chi_N + (1 - s_blN) * psiref;
    psi_errN = angdiff_custom(sN(5), h_refN);
    pos_errN = [sN(3)-xref; sN(4)-yref];
    vtanN = os_surge * sin(angdiff_custom(sN(5), chi_N));
    
    J = J + w_pos_T * (pos_errN' * pos_errN) + ...
            w_yaw_T * (psi_errN^2) + ...
            w_r_T * (sN(2)^2) + ...
            w_u_T * (uN^2) + ...
            w_tan_T * (vtanN^2);
end