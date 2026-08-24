%Refer to notes on derivatives
function dlogq = grad_theta_logq(theta,eta,mu,B,d,Transf,phi)
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

n = length(theta);
K = size(B,2);

Dm2 = sparse(1:n,1:n,1./(d.^2),n,n);

Sigma_phiinv = Dm2-Dm2*B/(sparse(1:K,1:K,1,K,K)+B'*Dm2*B)*B'*Dm2;  
switch Transf
    case ''
        dt = 1;
        ddt = 0;
    case 'YJ'
        dt = dYJ(theta,eta);
        ddt = ddYJ(theta,eta);
    case 'GH'
        dt = dgh(theta,eta(:,1),eta(:,2));
        ddt = ddgh(theta,eta(:,1),eta(:,2));
    case 'iGH'
        dt = digh(theta,eta(:,1),eta(:,2),phi);
        ddt = ddigh(theta,eta(:,1),eta(:,2),phi);
    case 'YJdouble'
        t2 = YJ(theta,eta(:,2));
        dt2 = dYJ(theta,eta(:,2));
        dt1 = dYJ(t2,eta(:,1));
        dt = dt1.*dt2;
        
        ddt2 = ddYJ(theta,eta(:,2));
        ddt1 =  ddYJ(t2,eta(:,1));
        ddt = dt1.*ddt2+(dt2.^2).*ddt1;
end
Tq1vec = ddt./dt;


dphidtheta = sparse(1:n,1:n,dt,n,n);
Tq2vec = -dphidtheta'*Sigma_phiinv*(phi-mu);

dlogq = Tq1vec+Tq2vec;
