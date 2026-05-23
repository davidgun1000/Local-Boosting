function [variance,MSE] = calculating_RKL(num_randeffect,num_param,mix_Mu,mix_T,mix_Weights,prior,num_period,group,y,x,num_some)
    
    num_samples = 500;
    num_component = length(mix_Mu);
    logMixWeights = log(mix_Weights);
    
    theta_G = mix_Mu{num_component,1}(num_randeffect+1:num_randeffect+num_param,1)+...
            ((mix_T{num_component,1}(num_randeffect+1:num_randeffect+num_param,num_randeffect+1:num_randeffect+num_param)')\randn(num_param,1));
    
    for k=1:num_component
         cholPrecs_G{k,1} = mix_T{k,1}(num_randeffect+1:num_randeffect+num_param,num_randeffect+1:num_randeffect+num_param)';
         logSampDensPerComp_G(k,1) = (-(num_param/2)*log(2*pi)+sum(log(diag(cholPrecs_G{k}))) ...
                -0.5*sum((cholPrecs_G{k}*(theta_G-(mix_Mu{k,1}(num_randeffect+1:num_randeffect+num_param,1))).^2)));
    end
    
    logRBindicator_G = logSampDensPerComp_G + logMixWeights';
    max_log_v_G = max(logRBindicator_G, [], 1);
    logTotalSampDens_G = log1p(sum(exp(logRBindicator_G - max_log_v_G),1)-1) + max_log_v_G;
    logRBindicator_G = logRBindicator_G - repmat(logTotalSampDens_G,num_component,1);
    mix_Weights_G = exp(logRBindicator_G)';
    logMixWeights_G = log(mix_Weights_G);
    beta = theta_G;
    id_alpha = (1:1:num_randeffect)';
    
    parfor j=1:num_randeffect
        
        [mean_cond_randeffect_j,chol_cond_randeffect_j] = calculate_condmeanvar_mixture(num_component,mix_Mu, mix_T,num_randeffect,num_param,j,theta_G);
        
        [thetaSampled_latent] = generate_samples(prior,mix_Mu,mix_T,mix_Weights,num_samples,num_some,j);
        
        [~, logSampDensPerComp_latent]=SampleFromMixture(logMixWeights_G,mean_cond_randeffect_j,chol_cond_randeffect_j,num_samples,0,thetaSampled_latent);
        [~,logTotalSampDens_latent]=CombineMixtureComponents(logMixWeights_G,logSampDensPerComp_latent);
         
        [log_posterior] = calculate_log_posterior(prior,num_samples,num_some,thetaSampled_latent,j,group,y,x,num_period,beta);
        
        log_weight_alpha = (log_posterior - logTotalSampDens_latent'); 
        max_alpha=max(real(log_weight_alpha));
        weight=real(exp(log_weight_alpha-max_alpha));
        weight=weight./sum(weight);
        variance(j,1) = var(log_weight_alpha);
        MSE(j,1) = mean((log_weight_alpha).^2);
    end
    
end
    
function [thetaSampled_latent] = generate_samples(prior,mix_Mu,mix_T,mix_Weights,num_samples,num_some,j)
    
    if j<=num_some

    for s=1:num_samples
        if rand<0.5
           thetaSampled_latent(1,s) = unifrnd(-2.5,-1.5); 
            
        else
           thetaSampled_latent(1,s) = unifrnd(1.5,2.5); 
        end        
    end

    else
        thetaSampled_latent(1,:) = linspace(-5,5,num_samples); 
        %thetaSampled_latent(1,:) = randn(1,num_samples);
    end

end

function [log_posterior] = calculate_log_posterior(prior,num_samples,num_some,thetaSampled_latent,j,group,y,x,num_period,beta)

    family = 'binomial';
    for s=1:num_samples
        if j<=num_some
            %[~, logZ, dlogZ, ~] = zpdf_general(thetaSampled_latent(:,s), prior.mean, prior.s, prior.a, prior.b);
            [logp_mixt, ~] = logpdf_deriv_mix2t(thetaSampled_latent(:,s), prior.w_a(1), prior.df, prior.mu(1), sqrt(prior.sig2_alpha(1)), prior.w_a(2), prior.df, ...
                prior.mu(2), sqrt(prior.sig2_alpha(2)));
            prior_alpha = logp_mixt;
            
        else
           prior_alpha = log(normpdf(thetaSampled_latent(:,s),0,1));    
        end
        id = group==j;
        eta = x(id,:)*beta+kron(thetaSampled_latent(:,s),ones(num_period,1));
        log_likelihood = y(id,1)'*eta-sum(b_fun(eta,family),1);
        log_posterior(s,1) = log_likelihood + prior_alpha;   
            
    end



end

    function [mean_cond_randeffect_j,chol_cond_randeffect_j] = calculate_condmeanvar_mixture(num_component,mix_Mu, mix_T,num_randeffect,num_param,j,theta_G)

        for k=1:num_component
            mean_cond_randeffect_j{k,1} = mix_Mu{k,1}(j,1) - (inv(mix_T{k,1}(j,j)))'*(mix_T{k,1}(num_randeffect+1:num_randeffect+num_param,j))'*...
                (theta_G - mix_Mu{k,1}(num_randeffect+1:num_randeffect+num_param,1));         
            chol_cond_randeffect_j{k,1} = mix_T{k,1}(j,j);
        end

    end
    

    
    
    


