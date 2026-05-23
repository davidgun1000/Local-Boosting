function [init_mu]=initialisation_global_parameter(num_randeffect,num_param,mix_Mu,mix_T,mix_Weights,indx,indx_diag,num_chosen,c,prior,num_period,group,y,x,num_some)

      num_samples=1000;
%     points = linspace(-2,2,num_points);
      psi_est = zeros(num_randeffect+num_param,1);
      num_tot = num_randeffect+num_param;
      num_component = length(mix_Mu);
      
      for j=1:num_component
          psi_est = psi_est + mix_Weights(j)*mix_Mu{j,1};
      end
%     
      for j=1:num_param
          thetaSampled_global = generate_samples(prior,num_samples);
          thetaSampled = [psi_est(1:num_randeffect+j-1,1)*ones(1,num_samples);thetaSampled_global;psi_est(num_randeffect+j+1:num_tot,1)*ones(1,num_samples)];
          [logTotSampDens_q] = compute_density_mixture(thetaSampled,num_tot,mix_Weights,mix_Mu,mix_T);
          for k=1:num_samples
              [log_posterior(k,1),~]=obtain_grad_logposterior_logit_some(y,x,thetaSampled(num_randeffect+1:num_randeffect+num_param,k),thetaSampled(1:num_randeffect,k),...
                     prior,num_period,group,num_some);
          end
          logw = log_posterior - logTotSampDens_q';
          max_logw=max(real(logw));
          weight=real(exp(logw-max_logw));
          weight=weight./sum(weight);
          indx = find(weight == max(weight));
          init_mu(j,1) = thetaSampled(num_randeffect+j,indx);
 
      end



end

function [thetaSampled_global] = generate_samples(prior,num_samples)
    

       %for s=1:num_samples 
           thetaSampled_global = sqrt(prior.sig2_beta)*randn(1,num_samples); 
       %end 

end