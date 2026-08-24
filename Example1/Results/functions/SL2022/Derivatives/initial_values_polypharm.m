function VBtransfobj = initial_values_polypharm(Transf,q,p)

VBtransfobj.mu = ones(q,1)*0.001;
VBtransfobj.tkappa = initial_values_tkappa(q,p);%Initial covariance matrix of q is the identity
VBtransfobj.tkappa(~tril(ones(size(VBtransfobj.tkappa))))  = -Inf;
VBtransfobj.l = ones(q,1)*(-0.5);

VBtransfobj.Transf = Transf;
switch Transf
    case 'YJ'
        VBtransfobj.eta = ones(q,1);
    case 'GH'
        VBtransfobj.eta = zeros(q,2);
        VBtransfobj.eta(:,1)=0.001;
        VBtransfobj.eta(:,2)=0.1;
    case ''
        VBtransfobj.eta = ones(q,1);
    case 'iGH'
        VBtransfobj.eta = zeros(q,2);
        VBtransfobj.eta(:,1)=0.00001;
        VBtransfobj.eta(:,2)=0.00001;
    case 'YJdouble'
        VBtransfobj.eta = ones(q,2)*1.3;
        VBtransfobj.eta(:,2) = 0.7;
end
