function [log_posterior,grad_log_posterior] = log_posterior_polypharm(y,x,theta,num_states,prior,dim_states,num_some)
   
   length_theta = length(theta); 
   lambda=(theta(length_theta,1));
   theta_states = theta(1:length_theta-1,1);
   [grad_lambda, grad_states, loglik_Z] = grad_ll_zrw(y, theta_states, lambda, prior.s, prior.a, prior.b, prior.hp_sig2, dim_states, x);
   grad_param = [grad_lambda];
   grad_log_posterior=[grad_states;grad_param];
   log_posterior=loglik_Z-((lambda(1,1)^2)/(2*prior.hp_sig2)); 

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










