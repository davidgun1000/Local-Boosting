function [LowerB,tkappa,mu,l,eta,ADA] = VB_step(tkappa, mu, l, eta, logpost,ADA,p,Transf)

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


oldEdelta2_tau = ADA.Edelta2_tau;
oldEg2_tau = ADA.Eg2_tau;

q = length(mu); %%% asssume each b_i is 1

zeps = randn(p+q,1)';
z = zeps(1:p)';
eps = zeps((p+1):end)';

tau = eta2tau(eta,Transf);
kappa = tkappa2kappa(tkappa);

[~,d,B] = kappa2bd(kappa);

phi = B*z + d.*eps;
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

[L_mu,L_tkappa,L_l,L_tau,g] = gradient_compute(theta,mu,kappa,tkappa,B,z,d,eps,tau,eta,logpost,Transf,phi,l);

L_tkappa(~tril(ones(size(L_tkappa))))  = 0;

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
%% Lowerbound
loghtheta = g;
kappa = tkappa2kappa(tkappa);
[~,d,B] = kappa2bd(kappa);
phi = B*z + d.*eps;

switch Transf
    case ''
        dt = 1.*(1./exp(l));
    case 'YJ'
        theta_tilde = iYJ(phi,eta);
        dt = dYJ(theta_tilde,eta).*(1./exp(l));
    case 'GH'
        theta_tilde = igh(phi,eta(:,1),eta(:,2));
        dt = dgh(theta_tilde,eta(:,1),eta(:,2)).*(1./exp(l));
    case 'iGH'
        theta_tilde = gh(phi,eta(:,1),eta(:,2));
        dt = digh(theta_tilde,eta(:,1),eta(:,2),phi).*(1./exp(l));
    case 'YJdouble'
        theta_tilde = iYJ(iYJ(phi,eta(:,1)),eta(:,2));
        t2 = YJ(theta_tilde,eta(:,2));
        dt2 = dYJ(theta_tilde,eta(:,2));
        dt1 = dYJ(t2,eta(:,1));
        dt = dt1.*dt2.*(1./exp(l));
end

%Sigma = B*B'+diag(d.^2);
%logphiNorm = logmvnpdf(phi',mu',Sigma);
logphiNorm = logmvnpdf2(B,z,d,eps);
logJacobian = sum(log(dt));
LowerB = loghtheta - logJacobian  - logphiNorm;




