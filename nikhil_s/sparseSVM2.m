function [a, E, a_hist] = sparseSVM2(data, y, K, eps, maxIter)
% Implementation of "Sparse learning for support vector classification" by Huang et al.
% 
% data: N x d matrix of training data
% y: N x 1 matrix
% K: N x N kernel matrix for training data
% eps: threshold for considering a point a support point

% weights on optimization terms:
C1 = 1;
C2 = 10;

N = size(data, 1);
a = ones(N,1); % SVM weights
E = 1/N*ones(N,1); % slack variables
lambda = ones(N,1); % objective function weights

% lower and upper bounds on SVM weights and slack variables
lb = [-inf(N,1); zeros(N,1)];
ub = inf(2*N,1);

a_hist = [];
options = optimoptions('quadprog', 'Display', 'none');
if nargin < 5, maxIter = 20; end

for iter = 1:maxIter
    display(sprintf('Iteration %d', iter));
    
    prev_nS = nnz(abs(a)>=eps);
    
    if iter == 1
        % L2 norm: sum(lambda_i * a_i^2), reduces to L0 norm:
        H = C1*diag([lambda; zeros(N,1)]);
        
        % L1 regularization: sum(E_i):
        f = C2*[zeros(N,1); ones(N,1)];
        
        % linear inequality constraint: y_i*K(x_i,.)*a + E >= 1 for all i
        A = -[bsxfun(@times, y, K), eye(N)];
        b = -ones(N,1);
        
        % use quadratic programming to find weights and slack variables
        a_eps = quadprog(H,f,A,b,[],[],lb,ub,[],options);
    else
        H(1:(2*N+1):2*N*N) = C1*lambda;
        a_eps = quadprog(H,f,A,b,[],[],lb,ub,a_eps,options);
    end
    
    a = a_eps(1:N);
    E = a_eps(N+1:end);
    
    % update lambda weights to approximate L0 norm on next iteration
    cond1 = find(abs(a) > eps);
    lambda(cond1) = 1./a(cond1).^2;
    cond2 = find(abs(a) <= eps);
    lambda(cond2) = 1./eps.^2;
    
    a_hist = [a_hist; a_eps'];
    
    nS = nnz(abs(a)>=eps)
    if nS == prev_nS && nS ~= N
        sprintf('Breaking early at iteration %d', iter)
        break;
    end
end
a(abs(a) < eps) = 0;
end
% function [a, E, a_hist] = sparseSVM2(data, y, K, eps, maxIter)
% % Implementation of "Sparse learning for support vector classification" by Huang et al.
% % 
% % data: N x d matrix of training data
% % y: N x 1 matrix
% % K: N x N kernel matrix for training data
% % eps: threshold for considering a point a support point
% %
% 
% % weights on optimization terms:
% C1 = 1;
% C2 = 10;
% 
% N = size(data, 1);
% a = ones(N,1); % SVM weights
% E = 1/N*ones(N,1); % slack variables
% lambda = ones(N,1); % objective function weights
% 
% % lower and upper bounds on SVM weights and slack variables
% lb = [-inf(N,1); zeros(N,1)];
% ub = inf(2*N,1);
% 
% a_hist = [];
% options = optimoptions('quadprog', 'Display', 'none');
% if nargin < 5, maxIter = 20; end
% 
% for iter = 1:maxIter
%     display(sprintf('Iteration %d', iter));
%     
%     prev_nS = nnz(abs(a)>=eps);
%     
%     if iter == 1
%         % L2 norm: sum(lambda_i * a_i^2), reduces to L0 norm:
%         H = C1*diag([lambda; zeros(N,1)]);
%         
%         % L1 regularization: sum(E_i):
%         f = C2*[zeros(N,1); ones(N,1)];
%         
%         % linear inequality constraint: y_i*K(x_i,.)*a + E >= 1 for all i
%         A = -[bsxfun(@times, y, K), eye(N)];
%         b = -ones(N,1);
%         
%         % use quadratic programming to find weights and slack variables
%         a_eps = quadprog(H,f,A,b,[],[],lb,ub,[],options);
%     else
%         H(1:(2*N+1):2*N*N) = C1*lambda;
%         a_eps = quadprog(H,f,A,b,[],[],lb,ub,a_eps,options);
%     end
%     
%     a = a_eps(1:N);
%     E = a_eps(N+1:end);
%     
%     % update lambda weights to approximate L0 norm on next iteration
%     cond1 = find(abs(a) > eps);
%     lambda(cond1) = 1./a(cond1).^2;
%     cond2 = find(abs(a) <= eps);
%     lambda(cond2) = 1./eps.^2;
%     
%     a_hist = [a_hist; a_eps'];
%     
%     nS = nnz(abs(a)>=eps)
%     if nS == prev_nS && nS ~= N
%         sprintf('Breaking early at iteration %d', iter)
%         break;
%     end
% end
% a(abs(a) < eps) = 0;
% end