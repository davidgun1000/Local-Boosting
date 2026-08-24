function pdfval = trans_dens(x,theta,Transf)

mu = theta(1);
sigma = theta(2);
switch Transf
    case ''
        phi = x;
        dt = 1;
    case 'YJ'
        eta = theta(3);
        phi = YJ(x,eta);
        dt = dYJ(x,eta);
    case 'iGH'
        eta1 = theta(3);
        eta2 = theta(4);
        phi = igh(x,eta1,eta2);
        dt = digh(x,eta1,eta2,phi);
    case 'YJdouble'
        eta1 = theta(3);
        eta2 = theta(4);
        
        t2 = YJ(x,eta2);
        dt2 = dYJ(x,eta2);

        phi = YJ(t2,eta1);
        dt1 = dYJ(t2,eta1);
        
        dt = abs(dt1.*dt2);
end
pdfval = exp(log_normpdf2(phi,mu,sigma)+log(dt));