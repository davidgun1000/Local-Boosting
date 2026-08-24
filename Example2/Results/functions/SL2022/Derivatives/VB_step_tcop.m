function [LowerB,tkappa,mu,l,eta,log_nu,ADA] = VB_step_tcop(tkappa, mu, l, eta, log_nu, logpost,ADA,p,Transf)

if strcmp(Transf,'')
    Transformation = 0;
else
    Transformation = 1;
end

rho = ADA.rho;
eps_step = ADA.eps_step;

oldEdelta2_mu = ADA.Edelta2_mu;
oldEg2_mu = ADA.Eg2_mu;

oldEdelta2_tkappa = ADA.Edelta2_tkappa;
oldEg2_tkappa = ADA.Eg2_tkappa;

oldEdelta2_l = ADA.Edelta2_l;
oldEg2_l = ADA.Eg2_l;

oldEdelta2_log_nu = ADA.Edelta2_log_nu;
oldEg2_log_nu = ADA.Eg2_log_nu;


oldEdelta2_tau = ADA.Edelta2_tau;
oldEg2_tau = ADA.Eg2_tau;

q = length(mu); %%% asssume each b_i is 1

J = 1;
L_mu_temp = 0;
L_tkappa_temp = 0;
L_l_temp =  0;
L_tau_temp = 0;
L_log_nu_temp = 0;
g_temp = 0;
for j = 1:J
    zeps = randn(p+q,1)';
    z = zeps(1:p)';
    eps = zeps((p+1):end)';
    u = rand;
    [w,dw] = dwdnu(u,log_nu2nu(log_nu));

    tau = eta2tau(eta,Transf);
    kappa = tkappa2kappa(tkappa);

    [~,d,B] = kappa2bd(kappa);

    phi = sqrt(w)*(B*z + d.*eps);
    switch Transf
        case ''
            theta = phi.*exp(l)+mu;
        case 'YJ'
            theta = iYJ(phi,eta).*exp(l)+mu;
        case 'GH'
            theta = igh(phi,eta(:,1),eta(:,2)).*exp(l)+mu;
        case 'iGH'
            theta = gh(phi,eta(:,1),eta(:,2)).*exp(l)+mu;
        case 'YJdouble'
            theta = iYJ(iYJ(phi,eta(:,1)),eta(:,2)).*exp(l)+mu;
    end

    [L_mu,L_tkappa,L_l,L_tau,L_log_nu,g] = gradient_compute_tcop(theta,mu,kappa,tkappa,B,z,d,eps,tau,eta,logpost,Transf,phi,l,log_nu,dw,w);
    L_tkappa(~tril(ones(size(L_tkappa))))  = 0;

    L_mu_temp     = L_mu + L_mu_temp;
    L_tkappa_temp = L_tkappa+L_tkappa_temp;
    L_l_temp      = L_l+L_l_temp;
    L_tau_temp    = L_tau+L_tau_temp;
    L_log_nu_temp = L_log_nu+L_log_nu_temp;
    g_temp         = g+g_temp;
end

L_mu = L_mu_temp/J;
L_tkappa = L_tkappa_temp/J;
L_l = L_l_temp/J;
L_tau = L_tau_temp/J;
L_log_nu = L_log_nu_temp/J;
g = g_temp/J;

%% mu update

ADA.Eg2_mu = rho*oldEg2_mu + (1-rho)*L_mu.^2;
Change_delta_mu = sqrt(oldEdelta2_mu + eps_step)./sqrt(ADA.Eg2_mu + eps_step).*L_mu;


mu = mu + Change_delta_mu;
ADA.Edelta2_mu = rho*oldEdelta2_mu + (1- rho)*Change_delta_mu.^2;


%% B update

vecL_tkappa = L_tkappa(:);

ADA.Eg2_tkappa = rho*oldEg2_tkappa + (1-rho)*vecL_tkappa.^2;
Change_delta_tkappa = sqrt(oldEdelta2_tkappa + eps_step)./sqrt(ADA.Eg2_tkappa + eps_step).*vecL_tkappa;


tkappa = tkappa + vec2mat(Change_delta_tkappa,q,p);
ADA.Edelta2_tkappa = rho*oldEdelta2_tkappa + (1- rho)*Change_delta_tkappa.^2;


%% d update

ADA.Eg2_l = rho*oldEg2_l + (1-rho)*L_l.^2;
Change_delta_l = sqrt(oldEdelta2_l + eps_step)./sqrt(ADA.Eg2_l + eps_step).*L_l;

l = l + Change_delta_l;
ADA.Edelta2_l = rho*oldEdelta2_l + (1- rho)*Change_delta_l.^2;

%% tau update
ADA.Eg2_tau = rho*oldEg2_tau + (1-rho)*L_tau.^2;
Change_delta_tau = sqrt(oldEdelta2_tau + eps_step)./sqrt(ADA.Eg2_tau + eps_step).*L_tau;
taustep = reshape(Change_delta_tau,size(tau));
tau = tau + taustep*Transformation;
ADA.Edelta2_tau = rho*oldEdelta2_tau + (1- rho)*Change_delta_tau.^2;
eta = tau2eta(tau,Transf);

%% nu update
ADA.Eg2_log_nu = rho*oldEg2_log_nu + (1-rho)*L_log_nu.^2;
Change_delta_log_nu = sqrt(oldEdelta2_log_nu + eps_step)./sqrt(ADA.Eg2_log_nu + eps_step).*L_log_nu;

log_nu = log_nu + Change_delta_log_nu;
ADA.Edelta2_log_nu = rho*oldEdelta2_log_nu + (1- rho)*Change_delta_log_nu.^2;
%% Lowerbound
loghtheta = g;
kappa = tkappa2kappa(tkappa);
[~,d,B] = kappa2bd(kappa);
phi = sqrt(w)*(B*z + d.*eps);

switch Transf
    case ''
        dt = 1./exp(l);
        logJacobian = sum(log(dt));
    case 'YJ'
        theta_tilde = iYJ(phi,eta);
        dt = dYJ(theta_tilde,eta).*(1./exp(l));
        logJacobian = sum(log(dt));
    case 'GH'
        theta_tilde = igh(phi,eta(:,1),eta(:,2));
        dt = dgh(theta_tilde,eta(:,1),eta(:,2)).*(1./exp(l));
        logJacobian = sum(log(dt));
    case 'iGH'
        theta_tilde = gh(phi,eta(:,1),eta(:,2));
        dt = digh(theta_tilde,eta(:,1),eta(:,2),phi).*(1./exp(l));
        logJacobian = sum(log(dt));
    case 'YJdouble'
        theta_tilde = iYJ(iYJ(phi,eta(:,1)),eta(:,2));
        t2 = YJ(theta_tilde,eta(:,2));
        dt2 = dYJ(theta_tilde,eta(:,2));
        dt1 = dYJ(t2,eta(:,1));
        dt = dt1.*dt2.*(1./exp(l));
        logJacobian = sum(log(dt));
end

nu = log_nu2nu(log_nu);
n = size(B,1);
K = size(B,2);
Dm2 = sparse(1:n,1:n,1./(d.^2),n,n);
Sigma_phiinv = Dm2-Dm2*B/(sparse(1:K,1:K,1,K,K)+B'*Dm2*B)*B'*Dm2; 
% 
log_detval = logdet(speye(K) + bsxfun(@times,B, 1./(d.^2))'*B) + sum(2*log(abs(d)));

Sigma_phiinvphi = Sigma_phiinv*phi;
phiSigma_phiinvphi = phi'*Sigma_phiinvphi;

logphit = gammaln(((nu+n)/2))-gammaln((nu/2))-(n/2)*log(nu*pi)-((nu+n)/2)*log(1+phiSigma_phiinvphi/nu)-0.5*log_detval;  


LowerB = loghtheta - logJacobian  - logphit;
if(LowerB==-Inf)
    disp(['LowerB = ' num2str(LowerB)]);
    disp(['loghtheta = ' num2str(loghtheta)]);
    disp(['logJacobian = ' num2str(logJacobian)]);
    disp(['logphit = ' num2str(logphit)]);
    disp(['phiSigma_phiinvphi = ' num2str(phiSigma_phiinvphi)]);
    disp(['log(det(Sigma_phiinv)) = ' num2str(log(det(Sigma_phiinv)))]);
    disp('B = ');
    disp(num2str(B));
    stop;
end



