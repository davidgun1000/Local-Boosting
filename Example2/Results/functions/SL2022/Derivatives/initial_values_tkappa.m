function tkappa = initial_values_tkappa(q,p)
d = ones(q,1);   %Starting the covariance matrix at the identity
%b = zeros(q,p)+0.001;
b = zeros(q,p)+0.5;
x = [d b];
if p>0
    kappa = x2kappa(x);
else
    kappa = zeros(q,p);
end
tkappa = kappa2tkappa(kappa);
