function [dB_dkappa,dd_dkappa] = dBd_dkappa(kappa)
[n,k] = size(kappa);
dXT_dkappa = dxT_dkappa(kappa);
In = sparse(1:n,1:n,1,n,n);


P1 = [sparse(k,1) sparse(1:k,1:k,1,k,k)]';
P2 = [1 sparse(1,k)]';
Kcom = communtationM(sparse(k+1,n));

dB_dXT = kron(P1',In)*Kcom;
dd_dXT = kron(P2',In)*Kcom;


dB_dkappa = dB_dXT*dXT_dkappa;
dd_dkappa = dd_dXT*dXT_dkappa;
