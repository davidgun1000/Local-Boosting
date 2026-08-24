function [w,dw] = dwdnu(u,nu)
method=1;
options = optimoptions('fmincon','Display','off');
if(method==1)
    fun = @(nu1) nu1./chi2inv(1-u,nu1); % w = F^{-1}_W(u;nu1)
    [~,w,~,~,~,dw,~] =  fmincon(fun,nu,[],[],[],[],nu,nu,[],options); %compute derivative
elseif(method==2)
    A=chi2inv(1-u,nu);
    fun = @(nu1) chi2inv(1-u,nu1); % w = F^{-1}_W(u;nu1)
    [~,x,~,~,~,dx,~] =  fmincon(fun,nu,[],[],[],[],nu,nu,[],options); %compute derivative of F_X^{-1} wrt nu
    dw=1/A-nu*dx/(A^2);
    w=nu./A;
else %this is the check case
    fun = @(nu1) nu1./chi2inv(1-u,nu1); % w = F^{-1}_W(u;nu1)
    [~,w,~,~,~,dw,~] =  fmincon(fun,nu,[],[],[],[],nu,nu,[],options); %compute derivative
    A=chi2inv(1-u,nu);
    fun2 = @(nu1) chi2inv(1-u,nu1); % w = F^{-1}_X(u;nu1)
    [~,x,~,~,~,dx,~] =  fmincon(fun2,nu,[],[],[],[],nu,nu,[],options); %compute derivative of F_X^{-1} wrt nu
    dw2=1/A-nu*dx/(A^2);
    w2=nu./A;
    if((abs(dw-dw2)/dw)>1.d-8)
        disp(['u = ' num2str(u)]);
        disp(['nu = ' num2str(nu)]);
        disp(['dw = ' num2str(dw)]);
        disp(['dw2 = ' num2str(dw2)]);
        disp(['(abs(dw-dw2)/dw)=' num2str(abs(dw-dw2)/dw)]);
        error('dw,dw2 different at dwdnu');
    end
end

