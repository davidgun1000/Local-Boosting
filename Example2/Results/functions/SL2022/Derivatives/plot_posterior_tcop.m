function theta = plot_posterior_tcop(ind,F_eta,F_mu,F_tkappa,F_log_nu,F_l,Transf,AKDE,varargin)
%Only first four posterior indicated by ind are plotted
if length(ind)>10
    warning('Only first ten posteriors will be displayed')
    ind = ind(1:10);
end


theta = VArand_tcop(10000,F_eta,F_mu,F_tkappa,F_l,F_log_nu,Transf);

for i = 1:length(ind)
    if length(ind)>1
        subplot(2,ceil(length(ind)/2),i)
    end
    if AKDE
        dist = fit_ssvkernel(theta(ind(i),:));
    else
        dist = paretotails(theta(ind(i),:),0,1,'Kernel');
    end
    lim = icdf(dist,[0.0001 1-0.0001]);
    llim = lim(1);
    ulim = lim(2);
    x = linspace(llim,ulim,100);
    plot(x,pdf(dist,x),varargin{:})
    ylabel(['f(\theta_{' mat2str(ind(i)) '}|y)'])
    xlabel(['\theta_{' mat2str(ind(i)) '}'])
end

