function dd_betat = ddbetat(theta,par1,par2,phi)
d_betat = dbetat(theta,par1,par2);
Phi_theta = normcdf(theta);
phi_theta = normpdf(theta);
phi_theta_prime = dnormpdf(theta,0,1,phi_theta);

phi_phi = normpdf(phi);
phi_phi_prime = dnormpdf(phi,0,1,phi_phi);

beta_Phi_theta = betapdf(Phi_theta,par1,par2);
beta_Phi_theta_prime = dbetapdf(Phi_theta,par1,par2,beta_Phi_theta);


dd_betat = (phi_phi.*((phi_theta.^2).*beta_Phi_theta_prime+beta_Phi_theta.*phi_theta_prime)-(phi_theta.*beta_Phi_theta.*phi_phi_prime.*d_betat))./(phi_phi.^2);
end


