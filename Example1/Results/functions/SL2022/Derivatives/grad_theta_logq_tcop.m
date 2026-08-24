%Refer to notes on derivatives
function [logq,dlogq] = grad_theta_logq_tcop(theta,phi,mu,l,eta,B,d,Transf,log_nu,temp)
if strcmp(Transf,'GH')
    if size(eta,2)~=2
        error('GH transformation requires 2 parameters. Make sure eta has two columns')
    end
end

if strcmp(Transf,'iGH')
    if size(eta,2)~=2
        error('iGH transformation requires 2 parameters. Make sure eta has two columns')
    end
end
K = size(B,2);
n = length(theta);

Dm2 = sparse(1:n,1:n,1./(d.^2),n,n);

Sigma_phiinv = Dm2-Dm2*B/(sparse(1:K,1:K,1,K,K)+B'*Dm2*B)*B'*Dm2; 
theta_tilde = (theta-mu)./exp(l);
dtheta_tildedtheta =1./exp(l);
switch Transf
    case ''
        ddt = 0;
        dt = 1;
        tfunc = @(thetatil) thetatil;
    case 'YJ'
        dt = dYJ(theta_tilde,eta);
        ddt = ddYJ(theta_tilde,eta);
        tfunc = @(thetatil) YJ(thetatil,eta);
    case 'GH'
        dt = dgh(theta_tilde,eta(:,1),eta(:,2));
        ddt = ddgh(theta_tilde,eta(:,1),eta(:,2));
        tfunc = @(thetatil) igh(thetatil,eta(:,1),eta(:,2));
    case 'iGH'
        dt = digh(theta_tilde,eta(:,1),eta(:,2),phi);
        ddt = ddigh(theta_tilde,eta(:,1),eta(:,2),phi);
        tfunc = @(thetatil) gh(thetatil,eta(:,1),eta(:,2));
    case 'YJdouble'
        t2 = YJ(theta_tilde,eta(:,2));
        dt2 = dYJ(theta_tilde,eta(:,2));
        dt1 = dYJ(t2,eta(:,1));
        dt = dt1.*dt2;
        
        ddt2 = ddYJ(theta_tilde,eta(:,2));
        ddt1 =  ddYJ(t2,eta(:,1));
        ddt = dt1.*ddt2+(dt2.^2).*ddt1;
end
Tq1vec = (ddt./dt)./exp(l);
Sigma_phiinvphi = Sigma_phiinv*phi;
phiSigma_phiinvphi = phi'*Sigma_phiinvphi;
nu = log_nu2nu(log_nu);
gprimeONg = -((nu+n)/(2*nu))*(1/(1+(phiSigma_phiinvphi/nu)));

dphidtheta = sparse(1:n,1:n,dt.*dtheta_tildedtheta,n,n);
Tq2vec = dphidtheta'*Sigma_phiinvphi*(2*gprimeONg);

dlogq = Tq1vec+Tq2vec;

if nargin == 10
    phi = tfunc(theta_tilde);
    Sigma_phiinvphi = Sigma_phiinv*phi;
    phiSigma_phiinvphi = phi'*Sigma_phiinvphi;
    logphit = gammaln(((nu+n)/2))-gammaln((nu/2))-(n/2)*log(nu*pi)-((nu+n)/2)*log(1+phiSigma_phiinvphi/nu);  
else
    %logphiNorm = logmvnpdf2(B,z,d,eps);                                %For computational speed
     logphit = [];
end

logJacobian = sum(log(dt)-l);
logq = logJacobian +logphit;




