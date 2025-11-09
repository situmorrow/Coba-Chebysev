function [c, ceq] = mpc_constraints_cheb(z, P, N, T, ship_params)
    % Nonlinear equality constraints: dynamics collocation & initial condition
    n_states = 5; n_controls = 1; n_nodes = N+1;
    S = reshape(z(1:n_states*n_nodes), n_states, n_nodes);
    U = reshape(z(n_states*n_nodes+1:end), n_controls, n_nodes);
    
    % Extract parameters
    x0 = P(1:n_states);
    
    % Precompute
    [taus, w, D] = cheb_nodes_weights_D(N);
    alpha = 2/T;
    
    % Equality constraints (collocation)
    ceq = [];
    for k = 1:n_nodes
        dS_k = zeros(n_states, 1);
        for i = 1:n_nodes
            dS_k = dS_k + D(k,i) * S(:,i);
        end
        f_val = ship_dynamics(S(:,k), U(:,k), ship_params);
        ceq = [ceq; alpha * dS_k - f_val];
    end
    
    % Initial condition
    ceq = [ceq; S(:,1) - x0];
    
    % No inequality constraints (handled by bounds & linear constraints)
    c = [];
end