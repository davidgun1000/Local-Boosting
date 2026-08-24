function theta = VArand_tcop(S,F_eta,F_mu,F_tkappa,F_l,F_log_nu,Transf)
F_kappa = tkappa2kappa(F_tkappa);
[~,F_d,F_B] = kappa2bd(F_kappa);

if strcmp(Transf,'GH')
    if size(F_eta,2)~=2
       error('GH transformation requires 2 parameters. Make sure eta has two columns') 
    end
end

if strcmp(Transf,'iGH')
    if size(F_eta,2)~=2
       error('GH transformation requires 2 parameters. Make sure eta has two columns') 
    end
end

n = size(F_B,1);
K = size(F_B,2);
z = randn(K,S);
epsilon = randn(n,S);
u = rand(1,S);
w = log_nu2nu(F_log_nu)./chi2inv(1-u,log_nu2nu(F_log_nu));
phi = sqrt(w).*(F_B*z+repmat(F_d,1,S).*epsilon);
switch Transf
    case ''
        theta = phi.*exp(F_l)+F_mu;
    case 'YJ'
        theta = iYJ(phi,repmat(F_eta,1,S)).*exp(F_l)+F_mu;
    case 'GH'
        Theta = igh(phi,repmat(F_eta(:,1),1,S),repmat(F_eta(:,2),1,S)).*exp(F_l)+F_mu;
        theta = reshape(Theta,size(F_eta,1),S);
    case 'iGH'
        Theta = gh(phi,repmat(F_eta(:,1),1,S),repmat(F_eta(:,2),1,S)).*exp(F_l)+F_mu;
        theta = reshape(Theta,size(F_eta,1),S);
    case 'YJdouble'
        theta = iYJ(iYJ(phi,repmat(F_eta(:,1),1,S)),repmat(F_eta(:,2),1,S)).*exp(F_l)+F_mu;
end

