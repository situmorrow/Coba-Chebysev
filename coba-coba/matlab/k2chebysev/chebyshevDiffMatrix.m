function D = chebyshevDiffMatrix(tau)
    % Compute Chebyshev differentiation matrix D for CGL nodes
    % D_ij ≈ d(phi_j)/dtau at tau_i
    
    N = length(tau) - 1;  % Polynomial order
    D = zeros(N+1, N+1);
    
    % Coefficients c_i: c_0 = c_N = 2, else c_i = 1
    c = ones(N+1, 1);
    c(1) = 2;
    c(end) = 2;
    
    % Compute matrix elements
    for i = 1:N+1
        for j = 1:N+1
            if i ~= j
                % Off-diagonal: D_ij = c_i*(-1)^(i+j) / (c_j*(tau_i-tau_j))
                D(i,j) = (c(i) * (-1)^(i+j)) / (c(j) * (tau(i) - tau(j)));
            else
                % Diagonal elements
                if i == 1  % i = 0 (tau = -1)
                    D(i,i) = (2*N^2 + 1) / 6;
                elseif i == N+1  % i = N (tau = +1)
                    D(i,i) = -(2*N^2 + 1) / 6;
                else  % Interior points
                    D(i,i) = -tau(i) / (2 * (1 - tau(i)^2));
                end
            end
        end
    end
end