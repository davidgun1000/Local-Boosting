function theta = ibetat(phi,par1,par2)
Phi_phi = normcdf(phi);
betavar = betainv(Phi_phi,par1,par2);
theta = norminv(betavar);
end


