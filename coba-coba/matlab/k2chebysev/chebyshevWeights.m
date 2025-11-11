function w = chebyshevWeights(N)
    % Compute Chebyshev-Gauss-Lobatto quadrature weights
    % Formula from PDF section 1.4.7
    
    w = zeros(N+1, 1);
    
    if mod(N, 2) == 0  % N even
        w(1) = 1/(N^2 - 1);
        w(N+1) = w(1);
        
        for k = 2:N  % k = 1,...,N-1 (0-based)
            sum_val = gammaFunc(0, k-1, N) + gammaFunc(N/2, k-1, N);
            for i = 1:(N/2 - 1)
                sum_val = sum_val + 2 * gammaFunc(i, k-1, N);
            end
            w(k) = (2/N) * sum_val;
        end
    else  % N odd
        w(1) = 1/N^2;
        w(N+1) = w(1);
        
        for k = 2:N  % k = 1,...,N-1 (0-based)
            sum_val = gammaFunc(0, k-1, N);
            for i = 1:((N-1)/2)
                sum_val = sum_val + 2 * gammaFunc(i, k-1, N);
            end
            w(k) = (2/N) * sum_val;
        end
    end
end

function val = gammaFunc(i, k, N)
    % Gamma function: gamma(i,k) = cos(2*pi*i*k/N) / (1-4*i^2)
    if i == 0
        val = 1;  % cos(0)/(1-0) = 1
    else
        val = cos(2 * pi * i * k / N) / (1 - 4 * i^2);
    end
end