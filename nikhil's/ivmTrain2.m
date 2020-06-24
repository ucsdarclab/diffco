function [a, S, idx] = ivmTrain2(X,y,K,lambda)
% [a, S, idx] = ivmTrain2(X,y,K,lambda) returns weight vector a, support set S,
% and indices of input data idx, given input data X, labels y, kernel
% matrix K, and regularization parameter lambda. Implementation of Zhu et
% al.'s Import Vector Machine. The last element of a is a constant bias
% term.

% B1
N = size(X, 1);
a = zeros(N+1,1);

S_mark = false(N,1);
R_mark = ~S_mark;

% B2
if nargin < 4, lambda = 1; end

Hk = inf(N,1);
for k = 1:N
    a_temp = cell(nnz(R_mark),1);
    H = zeros(N,1);
    
    % calculate all H costs for each point in R
    for l = find(R_mark)'
        idx = [find(S_mark); l];
        
        K_a = K(:, idx);
        K_q = K(idx, idx);
        
        F = K_a*a(idx) + a(end);
        p = 1./(1 + exp(-F));
        W = p.*(1-p);
        
        z = (F + (1./W).*(y-p));
        a_temp{l} = (K_a'*(W.*K_a) + lambda * K_q)\K_a'*(W.*z);
        
        F_temp = K_a*a_temp{l};
        H(l) = -y'*F_temp + sum(log(1+exp(F_temp))) + lambda/2*a_temp{l}'*K_q*a_temp{l};
    end
    
    % B3
    l = find(R_mark);
    [~, xls] = min(H(l));
    xls = l(xls);
    
    Hk(k) = H(xls);
    
    % calculate bias term
%     a(end) = mean(y([find(S_mark); xls]) - K([find(S_mark); xls],:)*a(1:N));
%     a(end) = -mean(K([find(S_mark); xls],:)*a(1:N));
    a(end) = -0.5 * (mean(K(y~=0,:)*a(1:N)) + mean(K(y==0,:)*a(1:N)));
%     a(end) = -mean(K*a(1:N));
    a([find(S_mark); xls]) = a_temp{xls};
    
    S_mark(xls) = true;
    R_mark(xls) = false;
    
    if k > 1 && abs((Hk(k) - Hk(k-1))/Hk(k)) < 0.001 % termination condition
        break;
    end
end
display(sprintf('Iterations: %d', k));

a = a([find(S_mark); N+1],:);
S = X(S_mark,:);
idx = find(S_mark);
end