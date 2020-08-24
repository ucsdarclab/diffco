function w = fitPolyharmonicSpline(c, h, k, lambda)
% c is N x d matrix of centers
% h is N x 1 vector of heights
% k is polyharmonic order

r = pdist2(c,c);
[N,d] = size(c);
if mod(k,2) % odd
    phi = r.^k;
    if nargin == 4
        phi(1:N+1:end) = lambda;
    end
else % even
    phi = r.^k.*log(r);
    if nargin < 4
        phi(1:N+1:end) = 0;
    else
        phi(1:N+1:end) = lambda;
    end
end

L = [phi c ones(N,1);  c' zeros(d,d+1);ones(1,N) zeros(1,d+1)];
w = linsolve(L,[h; zeros(d+1,1)]);
end