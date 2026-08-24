function f = b_fun(eta,family)
% compute the b-function at eta w.r.t. the family
if strcmp(family,'poisson') 
    f = exp(eta);
end
if strcmp(family,'binomial') 
    f = log(1+exp(eta));
end
