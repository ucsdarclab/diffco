function [a, originalIdx] = sparseSVM4(data, y, K, eps, maxIter)
% Implementation of "Sparse learning for support vector classification" by Huang et al.
% 
% data: N x d matrix of training data
% y: N x 1 matrix
% K: N x N kernel matrix for training data
% eps: threshold for considering a point a support point
%

% weights on optimization terms:
% C1 = 1;
% C = 1e5;
C = 100;

N = size(data, 1);
a = ones(N+1,1); % SVM weights
B = zeros(N,1);
nS = N;
originalIdx = 1:N+1;

K_bar = [ones(1, size(K,2)); K].*y';

% lower and upper bounds on SVM weights and slack variables
lb = zeros(N,1);
ub = C*ones(N,1);

f = -ones(N,1);

options = optimoptions('quadprog', 'Display', 'none');
if nargin < 5, maxIter = 20; end

for iter = 1:maxIter
    display(sprintf('Iteration %d', iter));
    
    lambda_invK = a.*K_bar;
    H = lambda_invK'*lambda_invK;
    
%     if iter == 1
%         B = quadprog(H,f,[],[],[],[],lb,ub,[],options);
%     else
%         B = quadprog(H,f,[],[],[],[],lb,ub,B,options);
%     end
    B = modSMO(B,H,C,1e6);
    
%     prev_a = a;
    a = diag(a.^2)*K_bar*B;
    a(abs(a)<=eps) = 0;
	
%     a_diff_norm = norm(a-prev_a);
    
    prev_nS = nS;
    nS = nnz(a(2:end))
    
    retainIdx = find(a);
    originalIdx = originalIdx(retainIdx);
    a = a(retainIdx);
%     fprintf('norm(a) = %f\n', norm(a));
    K_bar = K_bar(retainIdx, :);

    if nS == prev_nS && nS ~= N %a_diff_norm < 1e-4
        sprintf('Breaking early at iteration %d', iter)
        break;
    end
end
% a(abs(a) < eps) = 0;
end