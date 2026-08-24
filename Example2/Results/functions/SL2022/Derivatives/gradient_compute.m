function [L_mu,L_tkappa,L_l,L_tau,g] = gradient_compute(theta,mu,kappa,tkappa,B,z,d,eps,tau,eta,logpost,Transf,phi,l)
q = size(B,1);
p = size(B,2);
[g,delta_logh] = logpost(theta);

[~,delta_logq] = grad_theta_logq(theta,phi,mu,l,eta,B,d,Transf,z,eps);

L_mu = delta_logh - delta_logq;
if p>0
    dtheta_dtkappa = dthetadkappa(phi,kappa,z,eps,eta,l,Transf)*dkappa_dtkappa(tkappa);
else
    dtheta_dtkappa = zeros(q,p);
end
L_tkappa = reshape(dtheta_dtkappa'*(delta_logh - delta_logq),p,q)';
dtheta_dl = diag((theta-mu)./exp(l)).*exp(l);
L_l = dtheta_dl'*(delta_logh - delta_logq);
L_tau =  (dtheta_dtau(phi,tau,theta,Transf,eta,l)')*(delta_logh - delta_logq);    %% Derivatives with respect to tau, the fisher transformation of eta







