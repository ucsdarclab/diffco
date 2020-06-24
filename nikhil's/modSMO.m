function B = modSMO(B, H, C, maxIter)
tau = 0.001;
dQ = H*B - 1;
h = diag(H);

for iter = 1:maxIter
    cond = (dQ > tau & B > 0 | dQ < -tau & B < C);
    if any(cond)
        [~,i] = max((dQ.^2./h).*cond);
        B_prev = B(i);
        B(i) = min(max(0,B_prev-dQ(i)/h(i)),C);
        dQ = dQ + H(:,i)*(B(i)-B_prev);
    else
        'breaking'
        break;
    end
end
end