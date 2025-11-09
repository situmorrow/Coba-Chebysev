function d = angdiff_custom(a, b)
    % Selisih a - b dengan wrapping ke [-pi, pi]
    d = mod(a - b + pi, 2*pi) - pi;
end