function [L_mu,L_B,L_d,L_tau,g] = gradient_compute(theta,mu,B,z,d,eps,tau,eta,logpost,Transf,phi)
q = size(B,1);
p = size(B,2);
[g,delta_logh] = logpost(theta);

delta_logq = grad_theta_logq(theta,eta,mu,B,d,Transf,phi);
L_mu = dtheta_dmu(phi,eta,Transf,theta)*(delta_logh - delta_logq);
[dtheta_dB,dtheta_dd] = dtheta_dBDelta(eta,B,z,eps,Transf,phi,theta);
L_B = reshape(dtheta_dB'*(delta_logh - delta_logq),q,p);
L_d = dtheta_dd'*(delta_logh - delta_logq);
L_tau =  (dtheta_dtau(phi,tau,theta,Transf,eta)')*(delta_logh - delta_logq);    %% Derivatives with respect to tau, the fisher transformation of eta







