function [id_mean_update,init_mu,init_T] = initialisation_local_update_used(num_states,num_param,mix_Mu,mix_T,mix_Weights,indx,indx_diag,num_chosen,c,prior,y,num_some,...
    dim_states,x,indx_col)

    threshold_variance = 1000;
    num_samples = 500;
    num_component = length(mix_Mu)+1;
    
    [measure_latent] = calculating_RKL_higherdim(num_states,num_param,mix_Mu,mix_T,mix_Weights,prior,y,num_some,dim_states,x,indx_col);

    id_poor = measure_latent>=threshold_variance;
    length_id_poor = sum(id_poor);
    num_block = round(length_id_poor/num_chosen);
    if num_block ==0
       num_block=1; 
    end
    index_poor = find(id_poor == 1);
    
    if num_block==1
       index_poor_block{1,1} = index_poor; 
    else
       index_poor_block{1,1} = index_poor(1:num_chosen,1);
       for k=2:num_block-1
           index_poor_block{k,1} = index_poor((k-1)*num_chosen+1:k*num_chosen,1);
       end
       index_poor_block{num_block,1} = index_poor((num_block-1)*num_chosen+1:end,1);
    end
    which_block = randi(num_block);
    id_mean_update = index_poor_block{which_block,1};
    length_id_mean_update = length(id_mean_update);
    id_alpha = (1:1:num_states)';
    if length_id_mean_update == 0
       com = [id_alpha,measure_latent];
       com_sort = sortrows(com,2,'descend');
       id_mean_update = com_sort(1:num_chosen,1);  
        
    end
    
    
    grid_mu = [0];
    grid_T = [0.01,0.05,0.1];
    length_grid_mu = length(grid_mu);
    length_grid_T = length(grid_T);
    temp = [kron(grid_T',ones(length_grid_mu,1)),repmat(grid_mu',length_grid_T,1)];
    grid_com = [temp(:,2),temp(:,1)];    
    mix_Mu{num_component,1} = mix_Mu{num_component-1,1};
    mix_T{num_component,1} = mix_T{num_component-1,1};
    
    mix_Weights = [mix_Weights(1:num_component-2),0.5*mix_Weights(num_component-1),0.5*mix_Weights(num_component-1)];
    logMixWeights = log(mix_Weights);

    theta = mix_Mu{num_component,1}(1:dim_states*num_states+num_param,1)+...
            ((mix_T{num_component,1}(1:dim_states*num_states+num_param,1:dim_states*num_states+num_param)')\randn(dim_states*num_states+num_param,1));
    theta_G = theta(num_states+1:num_states+num_param,1);
    idx_beta1 = (1:dim_states:dim_states*num_states)';
    idx_beta2 = (2:dim_states:dim_states*num_states)';
    idx_beta3 = (3:dim_states:dim_states*num_states)';
    idx_beta4 = (4:dim_states:dim_states*num_states)';

    theta_states1 = theta(idx_beta1,1);
    theta_states2 = theta(idx_beta2,1);
    theta_states3 = theta(idx_beta3,1);
    theta_states4 = theta(idx_beta4,1);
    
    theta_states = theta(1:dim_states*num_states,1);

    for s1=1:length_grid_mu
        for s2=1:length_grid_T

        mix_Mu{num_component,1}(1:dim_states*num_states,1) = grid_mu(s1);
        for j=1:dim_states*num_states
            mix_T{num_component,1}(j,j) = grid_T(s2);
        end
        
        parfor j=1:length(id_mean_update)
            if id_mean_update(j,1)==num_states    

                [logSampDensPerComp_G] = obtain_logSampDensPerComp_G(mix_Mu,mix_T,num_states,num_param,theta_G,num_component,dim_states);
                logRBindicator_G = logSampDensPerComp_G + logMixWeights';
                max_log_v_G = max(logRBindicator_G, [], 1);
                logTotalSampDens_G = log1p(sum(exp(logRBindicator_G - max_log_v_G),1)-1) + max_log_v_G;
                logRBindicator_G = logRBindicator_G - repmat(logTotalSampDens_G,num_component,1);
                mix_Weights_G = exp(logRBindicator_G)';
                logMixWeights_G = log(mix_Weights_G);

                [mean_cond_randeffect_j,chol_cond_randeffect_j] = calculate_condmeanvar_mixture(num_component,mix_Mu, mix_T,num_states,num_param,...
                    id_mean_update(j,1),theta_G,dim_states,indx_col);
                [thetaSampled_latent1,thetaSampled_latent2,thetaSampled_latent3,thetaSampled_latent4] = generate_samples(prior,mix_Mu,mix_T,mix_Weights,num_samples,num_some,id_mean_update(j,1),dim_states);
                [~, logSampDensPerComp_latent]=SampleFromMixture(logMixWeights_G,mean_cond_randeffect_j,chol_cond_randeffect_j,num_samples,0,...
                    [thetaSampled_latent1;thetaSampled_latent2;thetaSampled_latent3;thetaSampled_latent4]);
                [~,logTotalSampDens_latent]=CombineMixtureComponents(logMixWeights_G,logSampDensPerComp_latent);
                [log_posterior] = calculate_log_posterior(prior,num_samples,num_some,[thetaSampled_latent1;thetaSampled_latent2;thetaSampled_latent3;thetaSampled_latent4],...
                    id_mean_update(j,1),y,theta_G,theta_states,num_component,mix_Mu,mix_T,num_states,num_param,logMixWeights,dim_states,x,indx_col);
                log_weight_alpha = (log_posterior - logTotalSampDens_latent');
                var_alpha{j,1}(s1,s2) = var(log_weight_alpha);
            else
                
                [logSampDensPerComp_G] = obtain_logSampDensPerComp_G(mix_Mu,mix_T,num_states,num_param,theta_G,num_component,dim_states);
                logRBindicator_G = logSampDensPerComp_G + logMixWeights';
                max_log_v_G = max(logRBindicator_G, [], 1);
                logTotalSampDens_G = log1p(sum(exp(logRBindicator_G - max_log_v_G),1)-1) + max_log_v_G;
                logRBindicator_G = logRBindicator_G - repmat(logTotalSampDens_G,num_component,1);
                mix_Weights_G = exp(logRBindicator_G)';
                logMixWeights_G = log(mix_Weights_G);


                [mean_cond_randeffect_condjplusone, chol_cond_randeffect_condjplusone]= calculate_condmeanvar_mixture_condplusone(num_component,mix_Mu,mix_T,num_states,num_param,id_mean_update(j,1),...
                    theta_G,[theta_states1(j+1,1);theta_states2(j+1,1);theta_states3(j+1,1);theta_states4(j+1,1)],dim_states,indx_col);
                [thetaSampled_latent1,thetaSampled_latent2,thetaSampled_latent3,thetaSampled_latent4] = generate_samples(prior,mix_Mu,mix_T,mix_Weights,num_samples,num_some,id_mean_update(j,1),dim_states);
                [~, logSampDensPerComp_latent_condplusone]=SampleFromMixture(logMixWeights_G,mean_cond_randeffect_condjplusone,chol_cond_randeffect_condjplusone,...
                    num_samples,0,[thetaSampled_latent1;thetaSampled_latent2;thetaSampled_latent3;thetaSampled_latent4]);
                [~,logTotalSampDens_latent_condplusone]=CombineMixtureComponents(logMixWeights_G,logSampDensPerComp_latent_condplusone);

                [log_posterior] = calculate_log_posterior(prior,num_samples,num_some,[thetaSampled_latent1;thetaSampled_latent2;thetaSampled_latent3;thetaSampled_latent4],id_mean_update(j,1),...
                        y,theta_G,theta_states,num_component,mix_Mu,mix_T,num_states,num_param,logMixWeights,dim_states,x,indx_col);
                log_weight_alpha = log_posterior - logTotalSampDens_latent_condplusone';
                var_alpha{j,1}(s1,s2) = var(log_weight_alpha);

            end
        end
    
        end
        
    end
    
    for j=1:length(id_mean_update)
        min_var_alpha(j,1) = min(min(var_alpha{j,1}));        
    end
        
    for j=1:length(id_mean_update)
        id_init = find(var_alpha{j,1} == min(min(var_alpha{j,1})));
        init_mu(dim_states*id_mean_update(j),1) = grid_com(id_init,1);
        init_mu(dim_states*id_mean_update(j)-1,1) = grid_com(id_init,1);
        init_mu(dim_states*id_mean_update(j)-2,1) = grid_com(id_init,1);
        init_mu(dim_states*id_mean_update(j)-3,1) = grid_com(id_init,1);
        
        init_T(indx_col.I(indx_col.par.Lii{id_mean_update(j),1}(1)),1) = grid_com(id_init,2);
        init_T(indx_col.I(indx_col.par.Lii{id_mean_update(j),1}(5)),1) = grid_com(id_init,2);
        init_T(indx_col.I(indx_col.par.Lii{id_mean_update(j),1}(8)),1) = grid_com(id_init,2);
        init_T(indx_col.I(indx_col.par.Lii{id_mean_update(j),1}(10)),1) = grid_com(id_init,2);
        
    end
    %grid_mu = [0];
    %grid_T = [1];

end
function [logSampDensPerComp_G] = obtain_logSampDensPerComp_G(mix_Mu,mix_T,num_states,num_param,theta_G,num_component,dim_states)

    for k=1:num_component
        cholPrecs_G{k,1} = mix_T{k,1}(dim_states*num_states+1:dim_states*num_states+num_param,dim_states*num_states+1:dim_states*num_states+num_param)';
        logSampDensPerComp_G(k,1) = (-(num_param/2)*log(2*pi)+sum(log(diag(cholPrecs_G{k}))) ...
               -0.5*sum((cholPrecs_G{k}*(theta_G-(mix_Mu{k,1}(dim_states*num_states+1:dim_states*num_states+num_param,1))).^2)));
    end

end



function [log_posterior] = calculate_log_posterior(prior,num_samples,num_some,thetaSampled_latent,...
    j,y,theta_G,theta_states,num_component,mix_Mu,mix_T,num_states,num_param,logMixWeights,dim_states,x,indx_col)

idx_beta1 = (1:dim_states:dim_states*num_states)';
idx_beta2 = (2:dim_states:dim_states*num_states)';
idx_beta3 = (3:dim_states:dim_states*num_states)';
idx_beta4 = (4:dim_states:dim_states*num_states)';


theta_states1 = theta_states(idx_beta1,1);
theta_states2 = theta_states(idx_beta2,1);
theta_states3 = theta_states(idx_beta3,1);
theta_states4 = theta_states(idx_beta4,1);


lambda = (theta_G(1,1));
v = exp(lambda);
T = length(theta_states1);
sig2 = prior.sig2;
if num_some<T

      if j==T
          for s=1:num_samples
              temp1 = -0.5*log(2*pi) - 0.5*lambda -0.5*(((y(j,1)-thetaSampled_latent(1,s) - thetaSampled_latent(2,s).*x(j,1) - thetaSampled_latent(3,s).*x(j,2) - thetaSampled_latent(4,s).*x(j,3)).^2)/v);
              temp2_beta1 = -0.5*log(2*pi) - 0.5*log(sig2) - 0.5.*(1/sig2).*((thetaSampled_latent(1,s)).^2);   
              temp2_beta2 = -0.5*log(2*pi) - 0.5*log(sig2) - 0.5.*(1/sig2).*((thetaSampled_latent(2,s)).^2);
              temp2_beta3 = -0.5*log(2*pi) - 0.5*log(sig2) - 0.5.*(1/sig2).*((thetaSampled_latent(3,s)).^2);
              temp2_beta4 = -0.5*log(2*pi) - 0.5*log(sig2) - 0.5.*(1/sig2).*((thetaSampled_latent(4,s)).^2);
                            
              log_posterior(s,1) = temp1+temp2_beta1+temp2_beta2+temp2_beta3+temp2_beta4;
          end
%         
       elseif j==1
          for s=1:num_samples
              temp1 = -0.5*log(2*pi) - 0.5*lambda -0.5*(((y(j,1)-thetaSampled_latent(1,s) - thetaSampled_latent(2,s).*x(j,1) - thetaSampled_latent(3,s).*x(j,2) - thetaSampled_latent(4,s).*x(j,3)).^2)/v);
              u1 = (theta_states1(j+1,1) - thetaSampled_latent(1,s))./ prior.s;
              u2 = (theta_states2(j+1,1) - thetaSampled_latent(2,s))./ prior.s;
              u3 = (theta_states3(j+1,1) - thetaSampled_latent(3,s))./ prior.s;
              u4 = (theta_states4(j+1,1) - thetaSampled_latent(4,s))./ prior.s;
              
              softplus = @(x) max(0,x) + log1p(exp(-abs(x)));
              logB1 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
              logB2 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
              logB3 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
              logB4 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
              
              temp2_beta1 = (-logB1 - log(prior.s) - prior.b*u1 - (prior.a+prior.b)*softplus(-u1));
              temp2_beta2 = (-logB2 - log(prior.s) - prior.b*u2 - (prior.a+prior.b)*softplus(-u2));
              temp2_beta3 = (-logB3 - log(prior.s) - prior.b*u3 - (prior.a+prior.b)*softplus(-u3));
              temp2_beta4 = (-logB4 - log(prior.s) - prior.b*u4 - (prior.a+prior.b)*softplus(-u4));
                            
              log_posterior(s,1) = temp1 + temp2_beta1 + temp2_beta2+temp2_beta3+temp2_beta4; 
              
           end
%          
      else
% %         
% %         
          for s=1:num_samples
              temp1 = -0.5*log(2*pi) - 0.5*lambda -0.5*(((y(j,1)-thetaSampled_latent(1,s) - thetaSampled_latent(2,s).*x(j,1) - thetaSampled_latent(3,s).*x(j,2) - thetaSampled_latent(4,s).*x(j,3)).^2)/v);           
              if j<=num_some
                 u1 = (theta_states1(j+1,1) - thetaSampled_latent(1,s))./ prior.s;
                 u2 = (theta_states2(j+1,1) - thetaSampled_latent(2,s))./ prior.s;
                 u3 = (theta_states3(j+1,1) - thetaSampled_latent(3,s))./ prior.s;
                 u4 = (theta_states4(j+1,1) - thetaSampled_latent(4,s))./ prior.s;
                 
                 softplus = @(x) max(0,x) + log1p(exp(-abs(x)));
                 logB1 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
                 logB2 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
                 logB3 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
                 logB4 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
                 
                 temp2_beta1 = (-logB1 - log(prior.s) - prior.b*u1 - (prior.a+prior.b)*softplus(-u1));
                 temp2_beta2 = (-logB2 - log(prior.s) - prior.b*u2 - (prior.a+prior.b)*softplus(-u2));
                 temp2_beta3 = (-logB3 - log(prior.s) - prior.b*u3 - (prior.a+prior.b)*softplus(-u3));
                 temp2_beta4 = (-logB4 - log(prior.s) - prior.b*u4 - (prior.a+prior.b)*softplus(-u4));
                 
                 
                 temp3 = calculate_states_statesmin1_Z(num_component,mix_Mu,mix_T,num_states,num_param,j,theta_G,[theta_states1(j+1,1);theta_states2(j+1,1);
                     theta_states3(j+1,1);theta_states4(j+1,1)],logMixWeights,num_samples,prior,[thetaSampled_latent(1,s);thetaSampled_latent(2,s);thetaSampled_latent(3,s);thetaSampled_latent(4,s)],...
                        num_some,dim_states,indx_col);
                 log_posterior(s,1) = temp1+temp2_beta1+temp2_beta2+temp2_beta3+temp2_beta4+temp3;
 
                           
             else
                  temp2_beta1 = -0.5*log(2*pi) - 0.5*log(sig2) - 0.5*(1./sig2).*((theta_states1(j+1,1) - thetaSampled_latent(1,s)).^2);
                  temp2_beta2 = -0.5*log(2*pi) - 0.5*log(sig2) - 0.5*(1./sig2).*((theta_states2(j+1,1) - thetaSampled_latent(2,s)).^2);
                  temp2_beta3 = -0.5*log(2*pi) - 0.5*log(sig2) - 0.5*(1./sig2).*((theta_states3(j+1,1) - thetaSampled_latent(3,s)).^2);
                  temp2_beta4 = -0.5*log(2*pi) - 0.5*log(sig2) - 0.5*(1./sig2).*((theta_states4(j+1,1) - thetaSampled_latent(4,s)).^2);
                  
                  temp3 = calculate_states_statesmin1_normal(num_component,mix_Mu,mix_T,num_states,num_param,j,theta_G,[theta_states1(j+1,1);theta_states2(j+1,1);
                      theta_states3(j+1,1);theta_states4(j+1,1)],...
                      logMixWeights,num_samples,prior,[thetaSampled_latent(1,s);thetaSampled_latent(2,s);thetaSampled_latent(3,s);thetaSampled_latent(4,s)],num_some,dim_states,indx_col);
                  log_posterior(s,1) = temp1+temp2_beta1+temp2_beta2+temp2_beta3+temp2_beta4+temp3;
             end
% %               
             
         end
% % %  
      end

    

else   
        
if j==T
   for s=1:num_samples
       softplus = @(x) max(0,x) + log1p(exp(-abs(x))); 
       temp1 = -0.5*log(2*pi) - 0.5*lambda -0.5*(((y(j,1)-thetaSampled_latent(1,s) - thetaSampled_latent(2,s).*x(j,1) - thetaSampled_latent(3,s).*x(j,2) - thetaSampled_latent(4,s).*x(j,3)).^2)/v);   
       u1 = thetaSampled_latent(1,s) ./ prior.s;
       u2 = thetaSampled_latent(2,s) ./ prior.s;
       u3 = thetaSampled_latent(3,s) ./ prior.s;
       u4 = thetaSampled_latent(4,s) ./ prior.s;
       
       logB1 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
       logB2 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
       logB3 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
       logB4 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
       
       
       temp2_beta1 = (-logB1 - log(prior.s) - prior.b*u1 - (prior.a+prior.b)*softplus(-u1));
       temp2_beta2 = (-logB2 - log(prior.s) - prior.b*u2 - (prior.a+prior.b)*softplus(-u2));
       temp2_beta3 = (-logB3 - log(prior.s) - prior.b*u3 - (prior.a+prior.b)*softplus(-u3));
       temp2_beta4 = (-logB4 - log(prior.s) - prior.b*u4 - (prior.a+prior.b)*softplus(-u4));
       
       
       log_posterior(s,1) = temp1 + temp2_beta1 + temp2_beta2+temp2_beta3+temp2_beta4;    
   end
 
elseif j==1

    for s=1:num_samples
         temp1 = -0.5*log(2*pi) - 0.5*lambda -0.5*(((y(j,1)-thetaSampled_latent(1,s) - thetaSampled_latent(2,s).*x(j,1) - thetaSampled_latent(3,s).*x(j,2) - thetaSampled_latent(4,s).*x(j,3)).^2)/v);  
         u1 = (theta_states1(j+1,1) - thetaSampled_latent(1,s))./ prior.s;
         u2 = (theta_states2(j+1,1) - thetaSampled_latent(2,s))./ prior.s;
         u3 = (theta_states3(j+1,1) - thetaSampled_latent(3,s))./ prior.s;
         u4 = (theta_states4(j+1,1) - thetaSampled_latent(4,s))./ prior.s;
         
         softplus = @(x) max(0,x) + log1p(exp(-abs(x)));
         logB1 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
         logB2 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
         logB3 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
         logB4 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
         
         temp2_beta1 = (-logB1 - log(prior.s) - prior.b*u1 - (prior.a+prior.b)*softplus(-u1));
         temp2_beta2 = (-logB2 - log(prior.s) - prior.b*u2 - (prior.a+prior.b)*softplus(-u2));
         temp2_beta3 = (-logB3 - log(prior.s) - prior.b*u3 - (prior.a+prior.b)*softplus(-u3));
         temp2_beta4 = (-logB4 - log(prior.s) - prior.b*u4 - (prior.a+prior.b)*softplus(-u4));
         
         log_posterior(s,1) = temp1 + temp2_beta1 + temp2_beta2 + temp2_beta3 + temp2_beta4; 
     end
 
 
else
%     
%     
%     
    for s=1:num_samples
        temp1 = -0.5*log(2*pi) - 0.5*lambda -0.5*(((y(j,1)-thetaSampled_latent(1,s) - thetaSampled_latent(2,s).*x(j,1) - thetaSampled_latent(3,s).*x(j,2) - thetaSampled_latent(4,s).*x(j,3)).^2)/v);   
        u1 = (theta_states1(j+1,1) - thetaSampled_latent(1,s))./ prior.s;
        u2 = (theta_states2(j+1,1) - thetaSampled_latent(2,s))./ prior.s;
        u3 = (theta_states3(j+1,1) - thetaSampled_latent(3,s))./ prior.s;
        u4 = (theta_states4(j+1,1) - thetaSampled_latent(4,s))./ prior.s;
        
        softplus = @(x) max(0,x) + log1p(exp(-abs(x)));
        logB1 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
        logB2 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
        logB3 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
        logB4 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
        
        temp2_beta1 = (-logB1 - log(prior.s) - prior.b*u1 - (prior.a+prior.b)*softplus(-u1));
        temp2_beta2 = (-logB2 - log(prior.s) - prior.b*u2 - (prior.a+prior.b)*softplus(-u2));
        temp2_beta3 = (-logB3 - log(prior.s) - prior.b*u3 - (prior.a+prior.b)*softplus(-u3));
        temp2_beta4 = (-logB4 - log(prior.s) - prior.b*u4 - (prior.a+prior.b)*softplus(-u4));
        
        temp3 = calculate_states_statesmin1_Z(num_component,mix_Mu,mix_T,num_states,num_param,j,theta_G,[theta_states1(j+1,1);theta_states2(j+1,1);
            theta_states3(j+1,1);theta_states4(j+1,1)],logMixWeights,num_samples,prior,[thetaSampled_latent(1,s);thetaSampled_latent(2,s);thetaSampled_latent(3,s);thetaSampled_latent(4,s)],...
            num_some,dim_states,indx_col);
        log_posterior(s,1) = temp1+temp2_beta1+temp2_beta2+temp2_beta3+temp2_beta4+temp3;
%          
     end
%  
end

end


end

function [temp_result] = calculate_states_statesmin1_Z(num_component,mix_Mu,mix_T,num_states,num_param,j,theta_G,theta_states,logMixWeights,num_samples,prior,thetaSampled_latent,num_some,dim_states,indx_col)
         
     lambda = (theta_G(1,1));    
     [mean_cond_randeffect_j,chol_cond_randeffect_j] = calculate_condmeanvar_mixture_condplusone(num_component,mix_Mu,mix_T,num_states,num_param,j,theta_G,theta_states,dim_states,indx_col);
      for k=1:num_component
          cholPrecs_G{k,1} = mix_T{k,1}(dim_states*num_states+1:dim_states*num_states+num_param,dim_states*num_states+1:dim_states*num_states+num_param)';
          logSampDensPerComp_G(k,1) = (-(num_param/2)*log(2*pi)+sum(log(diag(cholPrecs_G{k}))) ...
                   -0.5*sum((cholPrecs_G{k}*(theta_G-(mix_Mu{k,1}(dim_states*num_states+1:dim_states*num_states+num_param,1))).^2)));
          logSampDensPerComp_G_states_j(k,1) = logSampDensPerComp_G(k,1);
      end
      logRBindicator_G_states_j = logSampDensPerComp_G_states_j + logMixWeights';
      max_log_v_G_states_j = max(logRBindicator_G_states_j, [], 1);
      logTotalSampDens_G_states_j = log1p(sum(exp(logRBindicator_G_states_j - max_log_v_G_states_j),1)-1) + max_log_v_G_states_j;
      logRBindicator_G_states_j = logRBindicator_G_states_j - repmat(logTotalSampDens_G_states_j,num_component,1);
      mix_Weights_G_states_j = exp(logRBindicator_G_states_j)';
      logMixWeights_G_states_j = log(mix_Weights_G_states_j); 
      [theta_states_j, ~]=SampleFromMixture(logMixWeights_G_states_j,mean_cond_randeffect_j,chol_cond_randeffect_j,1,1,[]);

      [mean_cond_randeffect_jmin1,chol_cond_randeffect_jmin1] = calculate_condmeanvar_mixture_condplusone(num_component,mix_Mu,mix_T,num_states,num_param,j,theta_G,theta_states_j,dim_states,indx_col);
       for k=1:num_component
           cholPrecs_G{k,1} = mix_T{k,1}(num_states+1:num_states+num_param,num_states+1:num_states+num_param)';
           logSampDensPerComp_G(k,1) = (-(num_param/2)*log(2*pi)+sum(log(diag(cholPrecs_G{k}))) ...
                -0.5*sum((cholPrecs_G{k}*(theta_G-(mix_Mu{k,1}(num_states+1:num_states+num_param,1))).^2)));
           logSampDensPerComp_G_states_jmin1(k,1) = logSampDensPerComp_G(k,1);
       end
       logRBindicator_G_states_jmin1 = logSampDensPerComp_G_states_jmin1 + logMixWeights';
       max_log_v_G_states_jmin1 = max(logRBindicator_G_states_jmin1, [], 1);
       logTotalSampDens_G_states_jmin1 = log1p(sum(exp(logRBindicator_G_states_jmin1 - max_log_v_G_states_jmin1),1)-1) + max_log_v_G_states_jmin1;
       logRBindicator_G_states_jmin1 = logRBindicator_G_states_jmin1 - repmat(logTotalSampDens_G_states_jmin1,num_component,1);
       mix_Weights_G_states_jmin1 = exp(logRBindicator_G_states_jmin1)';
       logMixWeights_G_states_jmin1 = log(mix_Weights_G_states_jmin1); 
       [theta_states_jmin1, ~]=SampleFromMixture(logMixWeights_G_states_jmin1,mean_cond_randeffect_jmin1,chol_cond_randeffect_jmin1,1,1,[]);
        
       u1 = (thetaSampled_latent(1) - theta_states_jmin1(1))./ prior.s;
       softplus = @(x) max(0,x) + log1p(exp(-abs(x)));
       logB1 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
       temp1 = (-logB1 - log(prior.s) - prior.b*u1 - (prior.a+prior.b)*softplus(-u1));
       
       u2 = (thetaSampled_latent(2) - theta_states_jmin1(2))./ prior.s;
       softplus = @(x) max(0,x) + log1p(exp(-abs(x)));
       logB2 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
       temp2 = (-logB2 - log(prior.s) - prior.b*u2 - (prior.a+prior.b)*softplus(-u2));
       
       u3 = (thetaSampled_latent(3) - theta_states_jmin1(3))./ prior.s;
       softplus = @(x) max(0,x) + log1p(exp(-abs(x)));
       logB3 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
       temp3 = (-logB3 - log(prior.s) - prior.b*u3 - (prior.a+prior.b)*softplus(-u3));
       
       u4 = (thetaSampled_latent(4) - theta_states_jmin1(4))./ prior.s;
       softplus = @(x) max(0,x) + log1p(exp(-abs(x)));
       logB4 = gammaln(prior.a) + gammaln(prior.b) - gammaln(prior.a+prior.b);
       temp4 = (-logB4 - log(prior.s) - prior.b*u4 - (prior.a+prior.b)*softplus(-u4));
 
       
       
       temp_result = temp1+temp2+temp3+temp4;
       
       
            
         
end

function [temp_result] = calculate_states_statesmin1_normal(num_component,mix_Mu,mix_T,num_states,num_param,j,theta_G,theta_states,logMixWeights,num_samples,prior,thetaSampled_latent,num_some,dim_states,indx_col)
         
         lambda = (theta_G(1,1));
         [mean_cond_randeffect_j,chol_cond_randeffect_j] = calculate_condmeanvar_mixture_condplusone(num_component,mix_Mu,mix_T,num_states,num_param,j,theta_G,theta_states,dim_states,indx_col);
         for k=1:num_component
             cholPrecs_G{k,1} = mix_T{k,1}(num_states+1:num_states+num_param,num_states+1:num_states+num_param)';
             logSampDensPerComp_G(k,1) = (-(num_param/2)*log(2*pi)+sum(log(diag(cholPrecs_G{k}))) ...
                     -0.5*sum((cholPrecs_G{k}*(theta_G-(mix_Mu{k,1}(num_states+1:num_states+num_param,1))).^2)));
             logSampDensPerComp_G_states_j(k,1) = logSampDensPerComp_G(k,1);
         end
         logRBindicator_G_states_j = logSampDensPerComp_G_states_j + logMixWeights';
         max_log_v_G_states_j = max(logRBindicator_G_states_j, [], 1);
         logTotalSampDens_G_states_j = log1p(sum(exp(logRBindicator_G_states_j - max_log_v_G_states_j),1)-1) + max_log_v_G_states_j;
         logRBindicator_G_states_j = logRBindicator_G_states_j - repmat(logTotalSampDens_G_states_j,num_component,1);
         mix_Weights_G_states_j = exp(logRBindicator_G_states_j)';
         logMixWeights_G_states_j = log(mix_Weights_G_states_j); 
         [theta_states_j, ~]=SampleFromMixture(logMixWeights_G_states_j,mean_cond_randeffect_j,chol_cond_randeffect_j,1,1,[]);
         
         [mean_cond_randeffect_jmin1,chol_cond_randeffect_jmin1] = calculate_condmeanvar_mixture_condplusone(num_component,mix_Mu,mix_T,num_states,num_param,j,theta_G,theta_states_j,dim_states,indx_col);
         for k=1:num_component
             cholPrecs_G{k,1} = mix_T{k,1}(num_states+1:num_states+num_param,num_states+1:num_states+num_param)';
             logSampDensPerComp_G(k,1) = (-(num_param/2)*log(2*pi)+sum(log(diag(cholPrecs_G{k}))) ...
                     -0.5*sum((cholPrecs_G{k}*(theta_G-(mix_Mu{k,1}(num_states+1:num_states+num_param,1))).^2)));
             logSampDensPerComp_G_states_jmin1(k,1) = logSampDensPerComp_G(k,1);
         end
         logRBindicator_G_states_jmin1 = logSampDensPerComp_G_states_jmin1 + logMixWeights';
         max_log_v_G_states_jmin1 = max(logRBindicator_G_states_jmin1, [], 1);
         logTotalSampDens_G_states_jmin1 = log1p(sum(exp(logRBindicator_G_states_jmin1 - max_log_v_G_states_jmin1),1)-1) + max_log_v_G_states_jmin1;
         logRBindicator_G_states_jmin1 = logRBindicator_G_states_jmin1 - repmat(logTotalSampDens_G_states_jmin1,num_component,1);
         mix_Weights_G_states_jmin1 = exp(logRBindicator_G_states_jmin1)';
         logMixWeights_G_states_jmin1 = log(mix_Weights_G_states_jmin1); 
         [theta_states_jmin1, ~]=SampleFromMixture(logMixWeights_G_states_jmin1,mean_cond_randeffect_jmin1,chol_cond_randeffect_jmin1,1,1,[]);
         
         temp_result1 = -0.5*log(2*pi) - 0.5*log(prior.sig2) - 0.5*(1/(prior.sig2)).*((thetaSampled_latent(1) - theta_states_jmin1(1)).^2);
         temp_result2 = -0.5*log(2*pi) - 0.5*log(prior.sig2) - 0.5*(1/(prior.sig2)).*((thetaSampled_latent(2) - theta_states_jmin1(2)).^2);
         temp_result3 = -0.5*log(2*pi) - 0.5*log(prior.sig2) - 0.5*(1/(prior.sig2)).*((thetaSampled_latent(3) - theta_states_jmin1(3)).^2);
         temp_result4 = -0.5*log(2*pi) - 0.5*log(prior.sig2) - 0.5*(1/(prior.sig2)).*((thetaSampled_latent(4) - theta_states_jmin1(4)).^2);
         
         temp_result = temp_result1 + temp_result2 +temp_result3+temp_result4;
            
         
end



function [thetaSampled_latent1,thetaSampled_latent2,thetaSampled_latent3,thetaSampled_latent4] = generate_samples(prior,mix_Mu,mix_T,mix_Weights,num_samples,num_some,j,dim_states)

num_component=length(mix_Mu);
mean_1j = 0;
mean_2j = 0;
mean_3j = 0;
mean_4j = 0;
if j<=num_some
   for k=1:num_component
        mean_1j = mean_1j + mix_Weights(k)*mix_Mu{k,1}(dim_states*j-3,1);
        mean_2j = mean_2j + mix_Weights(k)*mix_Mu{k,1}(dim_states*j-2,1);
        mean_3j = mean_3j + mix_Weights(k)*mix_Mu{k,1}(dim_states*j-1,1);
        mean_4j = mean_4j + mix_Weights(k)*mix_Mu{k,1}(dim_states*j,1);
   end 
    
thetaSampled_latent1 = mean_1j+unifrnd(-10,10,1,num_samples);
thetaSampled_latent2 = mean_2j+unifrnd(-10,10,1,num_samples);
thetaSampled_latent3 = mean_3j+unifrnd(-10,10,1,num_samples);
thetaSampled_latent4 = mean_4j+unifrnd(-10,10,1,num_samples);

else
    
for k=1:num_component
    mean_1j = mean_1j + mix_Weights(k)*mix_Mu{k,1}(dim_states*j-3,1);
    mean_2j = mean_2j + mix_Weights(k)*mix_Mu{k,1}(dim_states*j-2,1);
    mean_3j = mean_3j + mix_Weights(k)*mix_Mu{k,1}(dim_states*j-1,1);
    mean_4j = mean_4j + mix_Weights(k)*mix_Mu{k,1}(dim_states*j,1);
end 
    
thetaSampled_latent1 = mean_1j+unifrnd(-3,3,1,num_samples);
thetaSampled_latent2 = mean_2j+unifrnd(-3,3,1,num_samples);
thetaSampled_latent3 = mean_3j+unifrnd(-3,3,1,num_samples);
thetaSampled_latent4 = mean_4j+unifrnd(-3,3,1,num_samples);
    
    
    
end
end

function [mean_cond_randeffect_j,chol_cond_randeffect_j] = calculate_condmeanvar_mixture(num_component,mix_Mu, mix_T,num_states,num_param,j,theta_G,dim_states,indx_col)
    
    

    for k=1:num_component
        mu_k = [mix_Mu{k,1}(dim_states*j-3,1);mix_Mu{k,1}(dim_states*j-2,1);mix_Mu{k,1}(dim_states*j-1,1);mix_Mu{k,1}(dim_states*j,1)];    
        chol_k(:,1) = [mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(1)),indx_col.J(indx_col.par.Lii{j,1}(1)));
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(2)),indx_col.J(indx_col.par.Lii{j,1}(2)));
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(3)),indx_col.J(indx_col.par.Lii{j,1}(3)));
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(4)),indx_col.J(indx_col.par.Lii{j,1}(4)))];
        chol_k(:,2) = [0;
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(5)),indx_col.J(indx_col.par.Lii{j,1}(5)));
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(6)),indx_col.J(indx_col.par.Lii{j,1}(6)));
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(7)),indx_col.J(indx_col.par.Lii{j,1}(7)))];
        chol_k(:,3) = [0;
                       0;
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(8)),indx_col.J(indx_col.par.Lii{j,1}(8)));
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(9)),indx_col.J(indx_col.par.Lii{j,1}(9)))];
        chol_k(:,4) = [0;           
                       0;
                       0;
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(10)),indx_col.J(indx_col.par.Lii{j,1}(10)))];
        
        chol_kG = [mix_T{k,1}(indx_col.I(indx_col.par.LGi{j,1}(1)),indx_col.J(indx_col.par.LGi{j,1}(1))),...
             mix_T{k,1}(indx_col.I(indx_col.par.LGi{j,1}(2)),indx_col.J(indx_col.par.LGi{j,1}(2))),...
             mix_T{k,1}(indx_col.I(indx_col.par.LGi{j,1}(3)),indx_col.J(indx_col.par.LGi{j,1}(3))),...
             mix_T{k,1}(indx_col.I(indx_col.par.LGi{j,1}(4)),indx_col.J(indx_col.par.LGi{j,1}(4)))];           
        
        
        mean_cond_randeffect_j{k,1} = mu_k - (inv(chol_k))'*(chol_kG)'*...
            (theta_G - mix_Mu{k,1}(dim_states*num_states+1:dim_states*num_states+num_param,1));         
        chol_cond_randeffect_j{k,1} = chol_k;
    end

end
%     
function [mean_cond_randeffect_j,chol_cond_randeffect_j] = calculate_condmeanvar_mixture_condplusone(num_component,mix_Mu,mix_T,num_states,num_param,j,theta_G,...
    theta_states_plusone,dim_states,indx_col)

    for k=1:num_component
         
        mu_k = [mix_Mu{k,1}(dim_states*j-3,1);mix_Mu{k,1}(dim_states*j-2,1);mix_Mu{k,1}(dim_states*j-1,1);mix_Mu{k,1}(dim_states*j,1)]; 
        mu_k_iplusone = [mix_Mu{k,1}(dim_states*(j+1)-3,1);mix_Mu{k,1}(dim_states*(j+1)-2,1);mix_Mu{k,1}(dim_states*(j+1)-1,1);mix_Mu{k,1}(dim_states*(j+1),1)];
        chol_k(:,1) = [mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(1)),indx_col.J(indx_col.par.Lii{j,1}(1)));
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(2)),indx_col.J(indx_col.par.Lii{j,1}(2)));
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(3)),indx_col.J(indx_col.par.Lii{j,1}(3)));
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(4)),indx_col.J(indx_col.par.Lii{j,1}(4)))];
        chol_k(:,2) = [0;
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(5)),indx_col.J(indx_col.par.Lii{j,1}(5)));
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(6)),indx_col.J(indx_col.par.Lii{j,1}(6)));
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(7)),indx_col.J(indx_col.par.Lii{j,1}(7)))];
        chol_k(:,3) = [0;
                       0;
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(8)),indx_col.J(indx_col.par.Lii{j,1}(8)));
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(9)),indx_col.J(indx_col.par.Lii{j,1}(9)))];
        chol_k(:,4) = [0;           
                       0;
                       0;
                       mix_T{k,1}(indx_col.I(indx_col.par.Lii{j,1}(10)),indx_col.J(indx_col.par.Lii{j,1}(10)))];
        
        chol_kG = [mix_T{k,1}(indx_col.I(indx_col.par.LGi{j,1}(1)),indx_col.J(indx_col.par.LGi{j,1}(1))),...
             mix_T{k,1}(indx_col.I(indx_col.par.LGi{j,1}(2)),indx_col.J(indx_col.par.LGi{j,1}(2))),...
             mix_T{k,1}(indx_col.I(indx_col.par.LGi{j,1}(3)),indx_col.J(indx_col.par.LGi{j,1}(3))),...
             mix_T{k,1}(indx_col.I(indx_col.par.LGi{j,1}(4)),indx_col.J(indx_col.par.LGi{j,1}(4)))];     
        
        chol_ktilde(:,1) = [mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(1)),indx_col.J(indx_col.par.Ltilde{j,1}(1)));
                            mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(2)),indx_col.J(indx_col.par.Ltilde{j,1}(2))); 
                            mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(3)),indx_col.J(indx_col.par.Ltilde{j,1}(3)));
                            mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(4)),indx_col.J(indx_col.par.Ltilde{j,1}(4)))];
                        
        chol_ktilde(:,2) = [mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(5)),indx_col.J(indx_col.par.Ltilde{j,1}(5)));
                            mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(6)),indx_col.J(indx_col.par.Ltilde{j,1}(6))); 
                            mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(7)),indx_col.J(indx_col.par.Ltilde{j,1}(7)));
                            mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(8)),indx_col.J(indx_col.par.Ltilde{j,1}(8)))];
                        
        chol_ktilde(:,3) = [mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(9)),indx_col.J(indx_col.par.Ltilde{j,1}(9)));
                            mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(10)),indx_col.J(indx_col.par.Ltilde{j,1}(10))); 
                            mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(11)),indx_col.J(indx_col.par.Ltilde{j,1}(11)));
                            mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(12)),indx_col.J(indx_col.par.Ltilde{j,1}(12)))];                
             
        chol_ktilde(:,4) = [mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(13)),indx_col.J(indx_col.par.Ltilde{j,1}(13)));
                            mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(14)),indx_col.J(indx_col.par.Ltilde{j,1}(14))); 
                            mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(15)),indx_col.J(indx_col.par.Ltilde{j,1}(15)));
                            mix_T{k,1}(indx_col.I(indx_col.par.Ltilde{j,1}(16)),indx_col.J(indx_col.par.Ltilde{j,1}(16)))];                
                        
        mean_cond_randeffect_j{k,1} = mu_k - (inv(chol_k))'*(chol_ktilde)'*(theta_states_plusone-mu_k_iplusone) - ...
             (inv(chol_k))'*(chol_kG)'*(theta_G - mix_Mu{k,1}(num_states+1:num_states+num_param,1));
        chol_cond_randeffect_j{k,1} = chol_k;
    end


end
