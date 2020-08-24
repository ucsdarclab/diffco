function d = modDiff(a,b,m)
    % performs a - b on a torus of circumference m in each dimension
    % example:
    % >> a = [2 2];
    % >> b = [-2 2];
    % >> a - b
    % ans =
    %      4     0
    % >> modDiff(a, b, 2*pi)
    % ans = 
    %   -2.2832         0
    d = mod(1.5*m+a-b, m) - m/2;
end