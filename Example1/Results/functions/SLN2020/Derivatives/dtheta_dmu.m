function dthetadmu = dtheta_dmu(phi,eta,Transf,theta)

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

dthetadmu=sparse(1:size(eta,1),1:size(eta,1),dit,size(eta,1),size(eta,1));