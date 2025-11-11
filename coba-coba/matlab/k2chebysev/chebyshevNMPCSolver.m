function [u_opt, predicted_states, info] = chebyshevNMPCSolver(...
    x0, reference, params, N, T, tau, D, w)
    
    % CHEBYSHEV NMPC SOLVER with atan2 orientation error
    % Uses fmincon from MATLAB Optimization Toolbox
    
    n_states = 5;
    n_controls = 1;
    total_nodes = N + 1;
    n_vars = (n_states + n_controls) * total_nodes;
    
    % Initial guess: linear interpolation
    states_guess = zeros(n_states, total_nodes);
    for i = 1:total_nodes
        progress = (i-1) / (total_nodes-1);
        guess_state = x0 + progress * ([0; 0; reference(1:2); reference(3)] - x0);
        states_guess(:,i) = guess_state;
    end
    controls_guess = zeros(1, total_nodes) + 0.01;
    
    z0 = [reshape(states_guess, n_states*total_nodes, 1); ...
          reshape(controls_guess, n_controls*total_nodes, 1)];
    
    % Variable bounds (only control bounds)
    lb = -inf(n_vars, 1);
    ub = inf(n_vars, 1);
    for k = 1:total_nodes
        idx = n_states*total_nodes + k;
        lb(idx) = params.u_min;
        ub(idx) = params.u_max;
    end
    
    % Optimization options
    options = optimoptions('fmincon', ...
        'Display', 'off', ...
        'MaxIterations', 300, ...
        'MaxFunctionEvaluations', 3000, ...
        'OptimalityTolerance', 1e-4, ...
        'ConstraintTolerance', 1e-4, ...
        'FiniteDifferenceType', 'central', ...
        'Algorithm', 'interior-point', ...
        'SpecifyObjectiveGradient', false, ...
        'SpecifyConstraintGradient', false);
    
    % Solve NLP
    try
        [z_opt, fval, exitflag, output] = fmincon(...
            @(z) objectiveFunction(z, reference, params, N, T, tau, D, w), ...
            z0, [], [], [], [], lb, ub, ...
            @(z) constraintFunction(z, x0, params, N, T, tau, D), ...
            options);
        
        % Extract solution
        states_opt = reshape(z_opt(1:n_states*total_nodes), n_states, total_nodes);
        controls_opt = reshape(z_opt(n_states*total_nodes+1:end), n_controls, total_nodes);
        
        u_opt = controls_opt(1,1);
        predicted_states = states_opt;
        
        info = struct('fval', fval, 'exitflag', exitflag, ...
            'iterations', output.iterations, 'solve_time', output.constrviolation);
    catch ME
        fprintf('Solver error: %s\n', ME.message);
        u_opt = 0;
        predicted_states = repmat(x0, 1, total_nodes);
        info = struct('fval', Inf, 'exitflag', -1, 'iterations', 0);
    end
end

function J = objectiveFunction(z, reference, params, N, T, tau, D, w)
    n_states = 5;
    n_controls = 1;
    total_nodes = N + 1;
    
    states = reshape(z(1:n_states*total_nodes), n_states, total_nodes);
    controls = reshape(z(n_states*total_nodes+1:end), n_controls, total_nodes);
    
    T_scale = T / 2;
    x_ref = reference(1);
    y_ref = reference(2);
    psi_ref = reference(3);
    
    J = 0;
    for k = 1:total_nodes
        % Position error
        pos_error = states(3:4,k) - [x_ref; y_ref];
        pos_cost = norm(pos_error)^2;
        
        % ========================================
        % MODIFIED: atan2 for orientation error
        % ========================================
        psi = states(5,k);
        orient_diff = atan2(sin(psi - psi_ref), cos(psi - psi_ref));
        orient_cost = orient_diff^2;
        
        % Control cost
        u = controls(1,k);
        control_cost = u^2;
        
        % Accumulate weighted cost
        J = J + w(k) * (params.w_position * pos_cost + ...
                        params.w_orientation * orient_cost + ...
                        params.w_control * control_cost);
    end
    
    J = T_scale * J;
end

function [c, ceq] = constraintFunction(z, x0, params, N, T, tau, D)
    n_states = 5;
    n_controls = 1;
    total_nodes = N + 1;
    
    states = reshape(z(1:n_states*total_nodes), n_states, total_nodes);
    controls = reshape(z(n_states*total_nodes+1:end), n_controls, total_nodes);
    
    T_scale = T / 2;
    
    % Equality constraints (dynamics + initial condition)
    ceq = [];
    
    % 1. Enforce dynamics at each node
    for k = 1:total_nodes
        s_dot_approx = D(k,:) * states';
        s_k = states(:,k);
        u_k = controls(:,k);
        f_val = shipModel(s_k, u_k, params);
        s_dot_actual = f_val * T_scale;
        ceq = [ceq; s_dot_approx' - s_dot_actual];
    end
    
    % 2. Initial condition
    ceq = [ceq; states(:,1) - x0];
    
    % Inequality constraints (state and control rates)
    c = [];
    
    % Yaw rate constraints: r_min <= r <= r_max
    for k = 1:total_nodes
        r = states(2,k);
        c = [c; params.r_min - r; r - params.r_max];
    end
    
    % Rudder rate constraints: |du/dt| <= rrot_max
    u_dot_tau = D * controls';
    for k = 2:total_nodes
        u_dot_t = (2/T) * u_dot_tau(k);
        c = [c; params.rrot_min - u_dot_t; u_dot_t - params.rrot_max];
    end
end