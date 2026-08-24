%This function provides the derivatives in Table 4 of
%Smith, Loaiza-Maya and Nott (2019) "High-dimensional Copula Variational Approximation through
%Transformation".
function [dthetadB,dthetadDelta] = dtheta_dBDelta(eta,B,z,epsilon,Transf_type,phi,theta)
%Common constants
n = size(B,1);
In = sparse(1:n,1:n,1,n,n);
TB1 = sparse(kron(z(:)',In));
TD1 = sparse(1:n,1:n,epsilon,n,n);

switch Transf_type
    case 'YJ'
        dit = diYJ(phi,eta);
    case 'GH'
        dit = digh(phi,eta(:,1),eta(:,2),theta);
    case 'iGH'
        dit = dgh(phi,eta(:,1),eta(:,2));
end

dthetadphi = sparse(1:n,1:n,dit,n,n);      %%Equivalent to dt(phi)/dphi
dthetadB = dthetadphi*TB1;
dthetadDelta = dthetadphi*(TD1);
