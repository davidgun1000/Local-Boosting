function [F_mu,F_l,F_tkappa,F_eta,F_LB,StoreTime] = VBtransf(logpost,q,p,Transf,niter,time_inter,do_plot,VBtransfobj)
if nargin == 7
    VBtransfobj = [];
end
if isempty(VBtransfobj)
    mu = ones(q,1)*0.001;
    tkappa = initial_values_tkappa(q,p);%Initial covariance matrix of q is the identity
    tkappa(~tril(ones(size(tkappa))))  = -Inf;
    l = ones(q,1)*(-5);
    switch Transf
        case 'YJ'
            eta = ones(q,1);
        case 'GH'
            eta = zeros(q,2);
            eta(:,1)=0.001;
            eta(:,2)=0.1;
        case ''
            eta = ones(q,1);
        case 'iGH'
            eta = zeros(q,2);
            eta(:,1)=0.00001;
            eta(:,2)=0.00001;
        case 'YJdouble'
            eta = ones(q,2)*1.3;
            eta(:,2) = 0.7;
    end
else
    tkappa = VBtransfobj.tkappa;
    mu = VBtransfobj.mu;
    l = VBtransfobj.l;
    eta = VBtransfobj.eta;
    Transf = VBtransfobj.Transf;
end
%%% ADADELTA Learning rate parameters
Edelta2_mu = zeros(length(mu),1);
Eg2_mu = zeros(length(mu),1);
Edelta2_tkappa = zeros(length(tkappa(:)),1);
Eg2_tkappa = zeros(length(tkappa(:)),1);
Edelta2_l = zeros(length(l),1);
Eg2_l = zeros(length(l),1);
Edelta2_eta = zeros(length(eta(:)),1);
Eg2_eta = zeros(length(eta(:)),1);
Edelta2_tau = zeros(length(eta(:)),1);
Eg2_tau = zeros(length(eta(:)),1);

ADA.rho = 0.95;
ADA.eps_step = 10^-6;
ADA.Edelta2_mu = Edelta2_mu;
ADA.Eg2_mu = Eg2_mu;
ADA.Edelta2_tkappa = Edelta2_tkappa;
ADA.Eg2_tkappa = Eg2_tkappa;
ADA.Edelta2_l = Edelta2_l;
ADA.Eg2_l = Eg2_l;
ADA.Edelta2_eta = Edelta2_eta;
ADA.Eg2_eta = Eg2_eta;
ADA.Edelta2_tau = Edelta2_tau;
ADA.Eg2_tau = Eg2_tau;
%%%----------------------------------------
StoreLB = zeros(niter,1);
temp = zeros(niter,1);
StoreTime = zeros(niter,1);

mu_sum = zeros(size(mu));
l_sum = zeros(size(l));
tkappa_sum = zeros(size(tkappa));
eta_sum = zeros(size(eta));
n_store = 1000;
if niter<n_store
    n_store = 1;
end
tic
for iter = 1:niter
    [LowerB,tkappa,mu,l,eta,ADA] = VB_step(tkappa, mu, l, eta, logpost,ADA,p,Transf);
    StoreLB(iter) = LowerB;
    temp(iter) = mu(end);
    StoreTime(iter) = toc;
    if ~(mod(iter,time_inter))
        disp(['Iter = ' mat2str(iter) '; LB = ' mat2str(round(StoreLB(iter)*100)/100) '; Time = ' mat2str(round(toc*100)/100)])
        if do_plot
            subplot(2,1,1)
            limit_lb=StoreLB(1:iter);
            plot(limit_lb(limit_lb>-1e6))
            subplot(2,1,2)
            plot(temp(1:iter))
            drawnow
        end
    end
    if iter >(niter-n_store)
        mu_sum = mu_sum + mu;
        l_sum = l_sum + l;
        tkappa_sum = tkappa_sum + tkappa;
        eta_sum = eta_sum + eta;
    end
end

F_mu = mu_sum/n_store;
F_l = l_sum/n_store;
F_tkappa = tkappa_sum/n_store;
F_eta = eta_sum/n_store;

F_LB = StoreLB;
