function [log_posterior, grad_log_posterior]=obtain_grad_param_statespace_sv_Z_higherdim(theta_G,theta_states,y,num_param,num_states,prior,num_some,dim_states, x)
   
   if num_some<num_states 
      lambda=(theta_G(1,1));
      v   = exp(lambda);  
      sig2 = prior.sig2;
      idx_beta1 = (1:dim_states:dim_states*num_states)';
      idx_beta2 = (2:dim_states:dim_states*num_states)';
      idx_beta3 = (3:dim_states:dim_states*num_states)';
      idx_beta4 = (4:dim_states:dim_states*num_states)';
      beta1 = theta_states(idx_beta1,1);
      beta2 = theta_states(idx_beta2,1);
      beta3 = theta_states(idx_beta3,1);
      beta4 = theta_states(idx_beta4,1);
      
      r   = y - beta1 - beta2.*x(:,1) - beta3.*x(:,2) - beta4.*x(:,3);   
      grad_lambda = 0.5*(sum(r.^2)./v - num_states)- lambda/prior.hp_sig2; 
      grad_beta1=zeros(num_states,1);
      grad_beta2=zeros(num_states,1);
      grad_beta3=zeros(num_states,1);
      grad_beta4=zeros(num_states,1);
      
      eta1 = beta1(2:num_some) - beta1(1:num_some-1);
      u1 = eta1 ./ prior.s;
      u1_initial = beta1(1) ./ prior.s;
      sigmoid1 = 0.5*(1 + tanh(u1./2));
      g1 = zeros(num_some,1);
      g1(2:end,1) = (1/prior.s) * (prior.a - (prior.a+prior.b).*sigmoid1);
      sigmoid1_initial = 0.5*(1 + tanh(u1_initial/2));
      g1(1,1) = (1/prior.s) * (prior.a - (prior.a+prior.b).*sigmoid1_initial);
      
      grad_beta1(1:num_some,1) = [ (r(1,1)./v)+g1(1)-g1(2); (r(2:num_some-1,1)./v)+(g1(2:num_some-1) - g1(3:num_some)); (r(num_some,1)./v) + g1(num_some,1) + (1/sig2)*...
           (beta1(num_some+1,1) - beta1(num_some,1))];
      grad_beta1(num_states,1) = (r(num_states,1)./v) + ...
              (-(1./sig2).*(beta1(num_states,1) - beta1(num_states-1,1)));
      grad_beta1(num_some+1:num_states-1,1) = (r(num_some+1:num_states-1,1)./v + ...
             (1/sig2).*(beta1(num_some+2:num_states,1) - beta1(num_some+1:num_states-1,1))  - ...
             (1/sig2).*(beta1(num_some+1:num_states-1,1) - beta1(num_some:num_states-2,1)));
      
      eta2 = beta2(2:num_some) - beta2(1:num_some-1);
      u2 = eta2 ./ prior.s;
      u2_initial = beta2(1) ./ prior.s;
      sigmoid2 =  0.5*(1 + tanh(u2./2));
      g2 = zeros(num_some,1);
      g2(2:end,1) = (1/prior.s) * (prior.a - (prior.a+prior.b).*sigmoid2);
      sigmoid2_initial = 0.5*(1 + tanh(u2_initial/2));
      g2(1,1) = (1/prior.s) * (prior.a - (prior.a+prior.b).*sigmoid2_initial);
      
      grad_beta2(1:num_some,1) = [ ((r(1,1).*x(1,1))./v)+g2(1)-g2(2); (r(2:num_some-1,1).*x(2:num_some-1,1)./v)+(g2(2:num_some-1) - g2(3:num_some)); (r(num_some,1).*x(num_some,1)./v) + g2(num_some,1) + (1/sig2)*...
           (beta2(num_some+1,1) - beta2(num_some,1))];
      grad_beta2(num_states,1) = (r(num_states,1).*x(num_states,1)./v) + ...
              (-(1./sig2).*(beta2(num_states,1) - beta2(num_states-1,1)));
      grad_beta2(num_some+1:num_states-1,1) = (r(num_some+1:num_states-1,1).*x(num_some+1:num_states-1,1)./v + ...
             (1/sig2).*(beta2(num_some+2:num_states,1) - beta2(num_some+1:num_states-1,1))  - ...
             (1/sig2).*(beta2(num_some+1:num_states-1,1) - beta2(num_some:num_states-2,1)));
         
      eta3 = beta3(2:num_some) - beta3(1:num_some-1);
      u3 = eta3 ./ prior.s;
      u3_initial = beta3(1) ./ prior.s;
      sigmoid3 =  0.5*(1 + tanh(u3./2));
      g3 = zeros(num_some,1);
      g3(2:end,1) = (1/prior.s) * (prior.a - (prior.a+prior.b).*sigmoid3);
      sigmoid3_initial = 0.5*(1 + tanh(u3_initial/2));
      g3(1,1) = (1/prior.s) * (prior.a - (prior.a+prior.b).*sigmoid3_initial);
      
      grad_beta3(1:num_some,1) = [ ((r(1,1).*x(1,2))./v)+g3(1)-g3(2); (r(2:num_some-1,1).*x(2:num_some-1,2)./v)+(g3(2:num_some-1) - g3(3:num_some)); (r(num_some,1).*x(num_some,2)./v) + g3(num_some,1) + (1/sig2)*...
           (beta3(num_some+1,1) - beta3(num_some,1))];
      grad_beta3(num_states,1) = (r(num_states,1).*x(num_states,2)./v) + ...
              (-(1./sig2).*(beta3(num_states,1) - beta3(num_states-1,1)));
      grad_beta3(num_some+1:num_states-1,1) = (r(num_some+1:num_states-1,1).*x(num_some+1:num_states-1,2)./v + ...
             (1/sig2).*(beta3(num_some+2:num_states,1) - beta3(num_some+1:num_states-1,1))  - ...
             (1/sig2).*(beta3(num_some+1:num_states-1,1) - beta3(num_some:num_states-2,1))); 
         
      eta4 = beta4(2:num_some) - beta4(1:num_some-1);
      u4 = eta4 ./ prior.s;
      u4_initial = beta4(1) ./ prior.s;
      sigmoid4 =  0.5*(1 + tanh(u4./2));
      g4 = zeros(num_some,1);
      g4(2:end,1) = (1/prior.s) * (prior.a - (prior.a+prior.b).*sigmoid4);
      sigmoid4_initial = 0.5*(1 + tanh(u4_initial/2));
      g4(1,1) = (1/prior.s) * (prior.a - (prior.a+prior.b).*sigmoid4_initial);
      
      grad_beta4(1:num_some,1) = [ ((r(1,1).*x(1,3))./v)+g4(1)-g4(2); (r(2:num_some-1,1).*x(2:num_some-1,3)./v)+(g4(2:num_some-1) - g4(3:num_some)); (r(num_some,1).*x(num_some,3)./v) + g4(num_some,1) + (1/sig2)*...
           (beta4(num_some+1,1) - beta4(num_some,1))];
      grad_beta4(num_states,1) = (r(num_states,1).*x(num_states,3)./v) + ...
              (-(1./sig2).*(beta4(num_states,1) - beta4(num_states-1,1)));
      grad_beta4(num_some+1:num_states-1,1) = (r(num_some+1:num_states-1,1).*x(num_some+1:num_states-1,3)./v + ...
             (1/sig2).*(beta4(num_some+2:num_states,1) - beta4(num_some+1:num_states-1,1))  - ...
             (1/sig2).*(beta4(num_some+1:num_states-1,1) - beta4(num_some:num_states-2,1)));      
         
      temp = [grad_beta1,grad_beta2,grad_beta3,grad_beta4]';
      grad_beta = temp(:);
      grad_log_posterior=[grad_beta;grad_lambda];

      softplus = @(x) max(0,x) + log1p(exp(-abs(x)));
      temp1 = -0.5*(num_states*log(2*pi) + num_states*lambda + sum(r.^2)/v);
  
      logB1 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b); 
      logB2 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b); 
      logB3 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b); 
      logB4 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b); 
      
      
      temp2_beta1 = (-logB1 - log(prior.s) - prior.b*(beta1(1,1)./prior.s) - (prior.a+prior.b)*softplus(-beta1(1,1)./prior.s)); 
      temp2_beta2 = (-logB2 - log(prior.s) - prior.b*(beta2(1,1)./prior.s) - (prior.a+prior.b)*softplus(-beta2(1,1)./prior.s));
      temp2_beta3 = (-logB3 - log(prior.s) - prior.b*(beta3(1,1)./prior.s) - (prior.a+prior.b)*softplus(-beta3(1,1)./prior.s));
      temp2_beta4 = (-logB4 - log(prior.s) - prior.b*(beta4(1,1)./prior.s) - (prior.a+prior.b)*softplus(-beta4(1,1)./prior.s));
      
      
      
      temp3_beta1_1 = sum(-logB1 - log(prior.s) - prior.b.*u1(1:num_some-1,1) - (prior.a+prior.b).*softplus(-u1(1:num_some-1,1)));
      temp3_beta1_2 = sum(-0.5*log(2*pi) - 0.5*log(sig2) - 0.5*(1./sig2).*((beta1(num_some+1:num_states,1) - ...
              beta1(num_some:num_states-1,1)).^2));

      temp3_beta2_1 = sum(-logB2 - log(prior.s) - prior.b.*u2(1:num_some-1,1) - (prior.a+prior.b).*softplus(-u2(1:num_some-1,1)));
      temp3_beta2_2 = sum(-0.5*log(2*pi) - 0.5*log(sig2) - 0.5*(1./sig2).*((beta2(num_some+1:num_states,1) - ...
              beta2(num_some:num_states-1,1)).^2));
      
      temp3_beta3_1 = sum(-logB3 - log(prior.s) - prior.b.*u3(1:num_some-1,1) - (prior.a+prior.b).*softplus(-u3(1:num_some-1,1)));
      temp3_beta3_2 = sum(-0.5*log(2*pi) - 0.5*log(sig2) - 0.5*(1./sig2).*((beta3(num_some+1:num_states,1) - ...
              beta3(num_some:num_states-1,1)).^2));    
      
      temp3_beta4_1 = sum(-logB4 - log(prior.s) - prior.b.*u4(1:num_some-1,1) - (prior.a+prior.b).*softplus(-u4(1:num_some-1,1)));
      temp3_beta4_2 = sum(-0.5*log(2*pi) - 0.5*log(sig2) - 0.5*(1./sig2).*((beta4(num_some+1:num_states,1) - ...
              beta4(num_some:num_states-1,1)).^2));    
          
      temp4=-((theta_G(1,1)^2)/(2*prior.hp_sig2));
      log_posterior=temp1+temp2_beta1+temp2_beta2+temp2_beta3+temp2_beta4+...
          temp3_beta1_1+temp3_beta1_2+temp3_beta2_1+temp3_beta2_2+temp3_beta3_1+temp3_beta3_2+temp3_beta4_1+temp3_beta4_2+temp4; 
   else
       
   lambda=(theta_G(1,1));
   [grad_lambda, grad_states, loglik_Z] = grad_ll_zrw(y, theta_states, lambda, prior.s, prior.a, prior.b, prior.hp_sig2, dim_states, x);
   grad_param = [grad_lambda];
   grad_log_posterior=[grad_states;grad_param];
   
   
   
   log_posterior=loglik_Z-((theta_G(1,1)^2)/(2*prior.hp_sig2));
   end
end   

function [grad_lambda, grad_beta, loglik_Z] = grad_ll_zrw(y, beta, lambda, s, a, b, sig2, dim_states, x)
%GRAD_LL_ZRW  Score wrt lambda=log(sigma^2) and all betas for RW+Z model.
%   [g_lambda, g_beta, loglik] = GRAD_LL_ZRW(y, beta, lambda, s, a, b)
%   Defaults: s=0.01, a=0.5, b=0.5 (horseshoe on logit scale).
%
%   y, beta : T x 1 vectors
%   lambda  : scalar  (lambda = log(sigma^2))
%   s,a,b   : Z-distribution scale and shape parameters
%
%   Returns:
%     grad_lambda : scalar  d/dlambda log p(y, beta | lambda)
%     grad_beta   : T x 1   d/dbeta_t log p(y, beta | lambda)
%     loglik      : scalar  joint log-density (up to any prior on beta1)
   
%     if nargin < 4 || isempty(s), s = 0.1; end
%     if nargin < 5 || isempty(a), a = 0.5;  end
%     if nargin < 6 || isempty(b), b = 0.5;  end

    T     = numel(y);

    % ----- observation part: y_t | beta_t ~ N(beta_t, sigma^2) -----
    v   = exp(lambda);
    
    idx_beta1 = (1:dim_states:dim_states*T)';
    idx_beta2 = (2:dim_states:dim_states*T)';
    idx_beta3 = (3:dim_states:dim_states*T)';
    idx_beta4 = (4:dim_states:dim_states*T)';
    beta1 = beta(idx_beta1,1);
    beta2 = beta(idx_beta2,1);
    beta3 = beta(idx_beta3,1);
    beta4 = beta(idx_beta4,1);
    r   = y - beta1 - beta2.*x(:,1) - beta3.*x(:,2) - beta4.*x(:,3);                    % residuals
    loglik_obs = -0.5*(T*log(2*pi) + T*lambda + sum(r.^2)/v);

    % d/dlambda log-lik: 0.5 * sum((r^2/v) - 1)
    grad_lambda = 0.5*(sum(r.^2)/v - T) - lambda/sig2;

    % contribution to d/dbeta_t from the Gaussian likelihood
    grad_beta1 = r./ v;
    grad_beta2 = (r.*x(:,1))./v;
    grad_beta3 = (r.*x(:,2))./v;
    grad_beta4 = (r.*x(:,3))./v;
    % ----- state increments: eta_t = beta_t - beta_{t-1} ~ Z(0, s, a, b) -----
    if T >= 2
       softplus = @(x) max(0,x) + log1p(exp(-abs(x))); 
       eta1 = beta1(2:end) - beta1(1:end-1);
       u1 = eta1 ./ s;
       u1_initial = beta1(1) ./ s;
       logB1 = gammaln(a) + gammaln(b) - gammaln(a+b); 
       loglik_rw1 = sum(-logB1 - log(s) - b*u1 - (a+b)*softplus(-u1));
       loglik_rw1_initial = -logB1 - b*u1_initial - (a+b)*softplus(-u1_initial);
       loglik_rw1_tot = loglik_rw1 + loglik_rw1_initial;
       sigmoid1 = 0.5*(1 + tanh(u1/2));
       sigmoid1_initial = 0.5*(1 + tanh(u1_initial/2));
       g1 = (1/s) * (a - (a+b).*sigmoid1);
       g1_initial = (1/s) * (a - (a+b).*sigmoid1_initial);
       inc1 = [ g1_initial-g1(1) ; g1(1:end-1) - g1(2:end) ; g1(end) ];
       
       eta2 = beta2(2:end) - beta2(1:end-1);
       u2 = eta2./ s;
       u2_initial = beta2(1)./s;
       logB2 =  gammaln(a) + gammaln(b) - gammaln(a+b); 
       loglik_rw2 = sum(-logB2 - log(s) - b*u2 - (a+b)*softplus(-u2));
       loglik_rw2_initial = -logB2 - b*u2_initial - (a+b)*softplus(-u2_initial); 
       loglik_rw2_tot = loglik_rw2 + loglik_rw2_initial;
       sigmoid2 = 0.5*(1 + tanh(u2/2));
       sigmoid2_initial = 0.5*(1 + tanh(u2_initial/2));
       g2 = (1/s) * (a - (a+b).*sigmoid2);
       g2_initial = (1/s) * (a - (a+b).*sigmoid2_initial);
       inc2 = [ g2_initial-g2(1) ; g2(1:end-1) - g2(2:end) ; g2(end) ];
       
       eta3 = beta3(2:end) - beta3(1:end-1);
       u3 = eta3./ s;
       u3_initial = beta3(1)./s;
       logB3 =  gammaln(a) + gammaln(b) - gammaln(a+b); 
       loglik_rw3 = sum(-logB3 - log(s) - b*u3 - (a+b)*softplus(-u3));
       loglik_rw3_initial = -logB3 - b*u3_initial - (a+b)*softplus(-u3_initial); 
       loglik_rw3_tot = loglik_rw3 + loglik_rw3_initial;
       sigmoid3 = 0.5*(1 + tanh(u3/2));
       sigmoid3_initial = 0.5*(1 + tanh(u3_initial/2));
       g3 = (1/s) * (a - (a+b).*sigmoid3);
       g3_initial = (1/s) * (a - (a+b).*sigmoid3_initial);
       inc3 = [ g3_initial-g3(1) ; g3(1:end-1) - g3(2:end) ; g3(end) ];
       
       eta4 = beta4(2:end) - beta4(1:end-1);
       u4 = eta4./ s;
       u4_initial = beta4(1)./s;
       logB4 =  gammaln(a) + gammaln(b) - gammaln(a+b); 
       loglik_rw4 = sum(-logB4 - log(s) - b*u4 - (a+b)*softplus(-u4));
       loglik_rw4_initial = -logB4 - b*u4_initial - (a+b)*softplus(-u4_initial); 
       loglik_rw4_tot = loglik_rw4 + loglik_rw4_initial;
       sigmoid4 = 0.5*(1 + tanh(u4/2));
       sigmoid4_initial = 0.5*(1 + tanh(u4_initial/2));
       g4 = (1/s) * (a - (a+b).*sigmoid4);
       g4_initial = (1/s) * (a - (a+b).*sigmoid4_initial);
       inc4 = [ g4_initial-g4(1) ; g4(1:end-1) - g4(2:end) ; g4(end) ];
       
    else
        loglik_rw1_tot = 0;
        loglik_rw2_tot = 0;
        loglik_rw3_tot = 0;
        loglik_rw4_tot = 0;
        
        inc1 = zeros(T,1);
        inc2 = zeros(T,1);
        inc3 = zeros(T,1);
        inc4 = zeros(T,1);
    end

    grad_beta1 = grad_beta1 + inc1;
    grad_beta2 = grad_beta2 + inc2;
    grad_beta3 = grad_beta3 + inc3;
    grad_beta4 = grad_beta4 + inc4;
    
    temp = [grad_beta1,grad_beta2,grad_beta3,grad_beta4]';
    grad_beta = temp(:);
    loglik_Z    = loglik_obs + loglik_rw1_tot + loglik_rw2_tot+loglik_rw3_tot+loglik_rw4_tot;
end