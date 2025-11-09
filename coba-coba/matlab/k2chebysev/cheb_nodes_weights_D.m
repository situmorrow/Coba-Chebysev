function [taus, w, D] = cheb_nodes_weights_D(N)
    % Nodes CGL: tau_k = cos(pi*k/N)
    k = (0:N)';
    taus = cos(pi * k / N);
    
    % Clenshaw-Curtis weights (integrasi)
    w = zeros(N+1, 1);
    if N == 0
        w = 2;
    else
        for j = 0:N
            s = 0;
            for m = 1:floor(N/2)
                s = s + (2/(1-4*m^2)) * cos(2*m*pi*j/N);
            end
            w(j+1) = (2/N) * (1 - s);
        end
    end
    
    % Differentiation matrix (Trefethen)
    c = [2; ones(N-1,1); 2] .* (-1).^k;
    X = repmat(taus, 1, N+1);
    dX = X - X.';
    D = (c * (1./c).') ./ (dX + eye(N+1));
    D = D - diag(sum(D,2));
end