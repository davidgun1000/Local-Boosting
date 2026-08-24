function [LowerB,B,mu,d,eta,ADA] = VB_step(B, mu, d, eta, logpost,ADA,p,Transf)
if strcmp(Transf,'')
    Transformation = 0;
else
    Transformation = 1;
end

rho = ADA.rho;
eps_step = ADA.eps_step;
oldEdelta2_mu = ADA.Edelta2_mu;
oldEg2_mu = ADA.Eg2_mu;

oldEdelta2_B = ADA.Edelta2_B;
oldEg2_B = ADA.Eg2_B;

oldEdelta2_d = ADA.Edelta2_d;
oldEg2_d = ADA.Eg2_d;


oldEdelta2_tau = ADA.Edelta2_tau;
oldEg2_tau = ADA.Eg2_tau;

q = length(mu); %%% asssume each b_i is 1

zeps = randn(p+q,1)';
z = zeps(1:p)';
eps = zeps((p+1):end)';

tau = eta2tau(eta,Transf);

phi = mu + B*z + d.*eps;
switch Transf
    case ''
        theta = phi;
    case 'YJ'
        theta = iYJ(phi,eta);
    case 'GH'
        theta = igh(phi,eta(:,1),eta(:,2));
    case 'iGH'
        theta = gh(phi,eta(:,1),eta(:,2));
    case 'YJdouble'
        theta = iYJ(iYJ(phi,eta(:,1)),eta(:,2));
end


[L_mu,L_B,L_d,L_tau,g] = gradient_compute(theta,mu,B,z,d,eps,tau,eta,logpost,Transf,phi);

L_B(~tril(ones(size(L_B))))  = 0;

%% mu update

ADA.Eg2_mu = rho*oldEg2_mu + (1-rho)*L_mu.^2;
Change_delta_mu = sqrt(oldEdelta2_mu + eps_step)./sqrt(ADA.Eg2_mu + eps_step).*L_mu;


mu = mu + Change_delta_mu;
ADA.Edelta2_mu = rho*oldEdelta2_mu + (1- rho)*Change_delta_mu.^2;


%% B update

vecL_B = L_B(:);

ADA.Eg2_B = rho*oldEg2_B + (1-rho)*vecL_B.^2;
Change_delta_B = sqrt(oldEdelta2_B + eps_step)./sqrt(ADA.Eg2_B + eps_step).*vecL_B;


B = B + vec2mat(Change_delta_B,q,p);
ADA.Edelta2_B = rho*oldEdelta2_B + (1- rho)*Change_delta_B.^2;


%% d update

ADA.Eg2_d = rho*oldEg2_d + (1-rho)*L_d.^2;
Change_delta_d = sqrt(oldEdelta2_d + eps_step)./sqrt(ADA.Eg2_d + eps_step).*L_d;

d = d + Change_delta_d;
ADA.Edelta2_d = rho*oldEdelta2_d + (1- rho)*Change_delta_d.^2;
%% tau update
ADA.Eg2_tau = rho*oldEg2_tau + (1-rho)*L_tau.^2;
Change_delta_tau = sqrt(oldEdelta2_tau + eps_step)./sqrt(ADA.Eg2_tau + eps_step).*L_tau;
taustep = reshape(Change_delta_tau,size(tau));
tau = tau + taustep*Transformation;
ADA.Edelta2_tau = rho*oldEdelta2_tau + (1- rho)*Change_delta_tau.^2;
eta = tau2eta(tau,Transf);
%% Lowerbound
loghtheta = g;
phi = mu + B*z + d.*eps;
switch Transf
    case ''
        dt = 1;
    case 'YJ'
        theta = iYJ(phi,eta);
        dt = dYJ(theta,eta);
    case 'GH'
        theta = igh(phi,eta(:,1),eta(:,2));
        dt = dgh(theta,eta(:,1),eta(:,2));
    case 'iGH'
        theta = gh(phi,eta(:,1),eta(:,2));
        dt = digh(theta,eta(:,1),eta(:,2),phi);
    case 'YJdouble'
        theta = iYJ(iYJ(phi,eta(:,1)),eta(:,2));
        t2 = YJ(theta,eta(:,2));
        dt2 = dYJ(theta,eta(:,2));
        dt1 = dYJ(t2,eta(:,1));
        dt = dt1.*dt2;
end
%Sigma = B*B'+diag(d.^2);
%logphiNorm = logmvnpdf(phi',mu',Sigma);
logphiNorm = logmvnpdf2(B,z,d,eps);
logJacobian = sum(log(dt));
LowerB = loghtheta - logJacobian  - logphiNorm;



