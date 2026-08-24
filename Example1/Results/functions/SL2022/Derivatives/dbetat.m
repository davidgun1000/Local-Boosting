function d_betat = dbetat(theta,par1,par2,phi)
Phi_theta = normcdf(theta);
if nargin ==3
    Beta_cdf = betacdf(Phi_theta,par1,par2);
    phi = norminv(Beta_cdf);
end
d_betat = (1./(normpdf(phi))).*betapdf(Phi_theta,par1,par2).*normpdf(theta);
end


