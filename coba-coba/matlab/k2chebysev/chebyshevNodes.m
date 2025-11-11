function tau = chebyshevNodes(N)
    % Generate Chebyshev-Gauss-Lobatto (CGL) nodes
    % N: polynomial order
    % Returns tau(1:N+1) nodes in [-1, 1]
    % Formula: tau_i = cos(i*pi/N), i = 0,...,N
    
    i = 0:N;
    tau = cos(i * pi / N);
end