function dthetadtau = dtheta_dtau(phi,tau,theta,Transf,eta)

if strcmp(Transf,'GH')
    if size(tau,2)~=2
       error('GH transformation requires 2 parameters. Make sure eta has two columns') 
    end
end

if strcmp(Transf,'iGH')
    if size(tau,2)~=2
       error('iGH transformation requires 2 parameters. Make sure eta has two columns') 
    end
end

q = size(tau,1);
detadtau = deta_dtau(tau,Transf);

switch Transf
    case ''
        dthetadtau = 0;
    case 'YJ'
        ditdeta = diYJ_deta(phi,eta);
        dthetadtau = sparse(1:q,1:q,ditdeta.*detadtau,q,q);
    case 'GH'
        dighdgh = digh(phi,eta(:,1),eta(:,2),theta);
        dghg = dgh_g(theta,eta(:,1),eta(:,2));
        dghh = dgh_h(theta,eta(:,1),eta(:,2));
        dghdeta = [dghg dghh];
        dthetadtau_unstacked = -repmat(dighdgh,1,2).*dghdeta.*detadtau;
        dthetadtau = sparse(1:q,1:q,dthetadtau_unstacked(:,1),q,q*2)+sparse(1:q,(q+1):q*2,dthetadtau_unstacked(:,2),q,q*2);
        %The negative value comes from the Triple Product Rule, or cyclic
        %chain rule
    case 'iGH'
        dghg = dgh_g(phi,eta(:,1),eta(:,2));
        dghh = dgh_h(phi,eta(:,1),eta(:,2));
        dghdeta = [dghg dghh];
        dthetadtau_unstacked = dghdeta.*detadtau;
        dthetadtau = sparse(1:q,1:q,dthetadtau_unstacked(:,1),q,q*2)+sparse(1:q,(q+1):q*2,dthetadtau_unstacked(:,2),q,q*2);
    case 'YJdouble'
         it1 = iYJ(phi,eta(:,1));
         dit2 = diYJ(it1,eta(:,2));
         ditdeta1 = dit2.*diYJ_deta(phi,eta(:,1));
         ditdeta2 = diYJ_deta(it1,eta(:,2));
         dthetadtau_unstacked = [ditdeta1 ditdeta2].*detadtau;
         dthetadtau = sparse(1:q,1:q,dthetadtau_unstacked(:,1),q,q*2)+sparse(1:q,(q+1):q*2,dthetadtau_unstacked(:,2),q,q*2);      
end

