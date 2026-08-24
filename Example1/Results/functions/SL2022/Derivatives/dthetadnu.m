function dtheta_dnu = dthetadnu(phi,w,dw,B,d,z,eps,eta,l,Transf)

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

switch Transf
    case ''
        dit = 1;
    case 'YJ'
        dit = diYJ(phi,eta);
    case 'GH'
        dit = digh(phi,eta(:,1),eta(:,2),theta);
    case 'iGH'
        dit = dgh(phi,eta(:,1),eta(:,2));
    case 'YJdouble'
        it1 = iYJ(phi,eta(:,1));
        dit1 = diYJ(phi,eta(:,1));
        dit2 = diYJ(it1,eta(:,2));
        dit = dit1.*dit2;
end
n = size(B,1);
dphi_dnu = dphidnu(w,B,z,d,eps,dw);
dtheta_dphi = sparse(1:n,1:n,dit.*exp(l),n,n);
dtheta_dnu = dtheta_dphi*dphi_dnu;