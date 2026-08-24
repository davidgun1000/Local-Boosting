function l_pdf = log_normpdf2(x,mu,sigma)
eps = 1e-10;
l_pdf = -0.5*log(2*pi)-0.5*log((sigma+eps).^2)-0.5*((x-mu).^2)./((sigma+eps).^2); 
