function [L_mu,L_tkappa,L_l,L_tau,L_log_nu,g] = gradient_compute_tcop(theta,mu,kappa,tkappa,B,z,d,eps,tau,eta,logpost,Transf,phi,l,log_nu,dw,w)
q = size(B,1);
p = size(B,2);
[g,delta_logh] = logpost(theta);

[~,delta_logq] = grad_theta_logq_tcop(theta,phi,mu,l,eta,B,d,Transf,log_nu);

L_mu = delta_logh - delta_logq;
if p>0
    dtheta_dtkappa = dthetadkappa(phi,kappa,z,eps,eta,l,Transf)*dkappa_dtkappa(tkappa);
else
    dtheta_dtkappa = zeros(q,p);
end
[~,dnu_dlog_nu] = log_nu2nu(log_nu);
dtheta_dlog_nu = dthetadnu(phi,w,dw,B,d,z,eps,eta,l,Transf)*dnu_dlog_nu;



L_tkappa = reshape(sqrt(w)*dtheta_dtkappa'*(delta_logh - delta_logq),p,q)';
dtheta_dl = diag((theta-mu)./exp(l)).*exp(l);
L_l = dtheta_dl'*(delta_logh - delta_logq);
L_tau =  (dtheta_dtau(phi,tau,theta,Transf,eta,l)')*(delta_logh - delta_logq);    %% Derivatives with respect to tau, the fisher transformation of eta
L_log_nu = dtheta_dlog_nu'*(delta_logh - delta_logq);







