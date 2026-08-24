function [log_posterior,grad_log_posterior]=obtain_grad_logposterior_logit_some(y,x,beta,alpha,prior,num_period,group,complex)


grad_prior_beta = -beta./prior.sig2_beta;
sig_alpha = sqrt(prior.sig2_alpha);
length_alpha = length(alpha);
alpha_complex = alpha(1:complex,1);
alpha_normal = alpha(complex+1:length_alpha,1);

[logp_mixt, dlogp_mixt] = logpdf_deriv_mix2t(alpha_complex, prior.w_a(1), prior.df, prior.mu(1), sig_alpha(1), prior.w_a(2), prior.df, prior.mu(2), sig_alpha(2));

grad_prior_alpha_complex = dlogp_mixt;
grad_prior_alpha_normal = -alpha_normal./prior.sig2_beta;
grad_prior_alpha = [grad_prior_alpha_complex;grad_prior_alpha_normal];
eta=x*beta+kron(alpha,ones(num_period,1));

for i=1:length(unique(group))
    id = group == i;
    grad_log_likelihood_alpha(i,1) = sum(y(id,1)-(exp(eta(id,1))./(1+exp(eta(id,1)))));
    
    
end

grad_log_likelihood_beta = x'*y(:,1)-x'*(exp(eta)./(1+exp(eta)));
grad_log_posterior=[grad_log_likelihood_alpha;grad_log_likelihood_beta] + [grad_prior_alpha;grad_prior_beta];%[zeros(ngroups,1);grad_beta] + [grad_alpha;zeros(length(beta),1)];

prior_beta= -beta'*beta/prior.sig2_beta/2;
prior_alpha_normal = -alpha_normal'*alpha_normal/prior.sig2_beta/2;
prior_alpha_complex = sum(logp_mixt);

Nn = length(y); % total number of observation
family = 'binomial';
llh_calc=y(:,1)'*eta-sum(b_fun(eta,family),1);
log_posterior=prior_alpha_normal + prior_alpha_complex + prior_beta + llh_calc;


end


% theta = [alpha;beta];
%     grad_beta = -beta./sig2_beta;
%     sig_alpha = sqrt(sig2_alpha);
%     cons = w_a(1)*normpdf(alpha,0,sig_alpha(1)) + w_a(2)*normpdf(alpha,0,sig_alpha(2));
%     grad_alpha    = -alpha./cons.*( w_a(1)*normpdf(alpha,0,sig_alpha(1))./sig2_alpha(1) + w_a(2)*normpdf(alpha,0,sig_alpha(2))./sig2_alpha(2) );
%     
%     %grad_com_first = grad_ll
%     %grad_com_second = grad of bias correction
%     
%     grad_com_third=[grad_alpha;grad_beta];%[zeros(ngroups,1);grad_beta] + [grad_alpha;zeros(length(beta),1)];