function dtheta = dibetat(phi,par1,par2)
Phi_phi = normcdf(phi);
betavar = betainv(Phi_phi,par1,par2);
theta = norminv(betavar);
dtheta = (1./normpdf(theta)).*(1./betapdf(betavar,par1,par2)).*normpdf(phi);
end
