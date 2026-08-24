%%%This function converts from angles to b and d
function [x,d,b] = kappa2bd(kappa)
r = 1;
[n,k] = size(kappa);
sinkappa = [ones(n,1) sin(kappa)];
coskappa = [cos(kappa) ones(n,1)];
sinkappacumprod = cumprod(sinkappa,2);
x = r*coskappa.*sinkappacumprod;
b = x(:,2:end);
d = x(:,1);
