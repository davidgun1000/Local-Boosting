function kappa = x2kappa(x)
cumsum_squared_inv = cumsum(x(:,end:(-1):1).^2,2);
sqrt_cumsum = sqrt(cumsum_squared_inv(:,end:(-1):1));
kappa = acos(x(:,1:end-1)./sqrt_cumsum(:,1:end-1));
kappa(x(:,end)<0,end) = 2*pi-kappa(x(:,end)<0,end);