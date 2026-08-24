function theta = VArand(S,F_eta,F_mu,F_B,F_d,Transf)

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

phi = F_mu + F_B*z+repmat(F_d,1,S).*epsilon;
switch Transf
    case ''
        theta = phi;
    case 'YJ'
        theta = iYJ(phi,repmat(F_eta,1,S));
    case 'GH'
        Theta = igh(phi,repmat(F_eta(:,1),1,S),repmat(F_eta(:,2),1,S));
        theta = reshape(Theta,size(F_eta,1),S);
    case 'iGH'
        Theta = gh(phi,repmat(F_eta(:,1),1,S),repmat(F_eta(:,2),1,S));
        theta = reshape(Theta,size(F_eta,1),S);
    case 'YJdouble'
        theta = iYJ(iYJ(phi,repmat(F_eta(:,1),1,S)),repmat(F_eta(:,2),1,S));
end
