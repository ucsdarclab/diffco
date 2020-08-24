function d = modDist(a,b,m)
    d = sqrt(sum((m/2 - mod(1.5*m+bsxfun(@minus,a,b),m)).^2,2));
end