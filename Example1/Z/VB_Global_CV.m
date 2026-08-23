%This code implements Global boosting for logistic regression model with Z
%distribution for the prior for random effects. Control variate method is
%used to reduce the variance of gradient of lower bound with respect to
%cholesky factor.
%parpool(10)
load('polypharm_data.mat');%load the dataset
num_param=size(data(:,3:end),2)+1; %number of global parameters
num_randeffect = length(unique(data(:,2)));% number of random effects
num_tot = num_randeffect+num_param;
num_period = length(data(:,1))/num_randeffect;
num_chosen = [];
mix_Mu{1,1}=zeros(num_tot,1); %initial values for mean of the first component
mix_T{1,1}=(100)*speye(num_tot); %initial values for the cholesky factor of the first component
mix_T_das{1,1}=log(100)*speye(num_tot);
indx = obtain_index_full_randeffect(num_randeffect,num_param); % finding index of the random effects
indx_diag(:,1) = (1:num_tot)';
indx_diag(:,2) = (1:num_tot)';






delta_T=speye(num_tot);
num_samples = 10; %number of samples
prior.sig2_beta = 1; %hyperparameters for the priors
prior.mean = 0;
prior.s = 0.1;
prior.a = 0.5;
prior.b = 0.5;

y = data(:,1); %response variable
group = data(:,2);

n = length(data(:,1));
x = [ones(n,1),data(:,3:end)]; %matrix of predictors

nrComponent = 2; %number of components

adapt_tau_1=0.9; %ADAM hyperparameters
adapt_tau_2=0.99;
adapt_epsilon=10^-8;
adapt_alpha_mu=0.01;
adapt_alpha_T = 0.001;
adapt_alpha_weight = 0.001;
m_t_mu{1,1}=0;
v_t_mu{1,1}=0;

m_t_T{1,1}=0;
v_t_T{1,1}=0;

m_t_weight = 0;
v_t_weight = 0;

mt_hat_weight = 0;
vt_hat_weight = 0;

mt_hat_T{1,1} = 0;
vt_hat_T{1,1} = 0;

mt_hat_mu{1,1} = 0;
vt_hat_mu{1,1} = 0;

adapt_grad_weight_mu = 0;
adapt_grad_weight_T = 0;
adapt_grad_weight_weight = 0;

max_iter = 5000; %number of iterations
mix_Weights = 1;
mix_Weights_actual=1;
num_some= num_randeffect; %the number of random effects with complex prior distributions

measure_latent = [];
var_alpha = [];
window = 100;
for c=1:nrComponent
    %c
    if c==1
    %optimisation for the first component using stochastic gradient ascent
    %with reparameterisation trick
    parfor s=1:num_samples
            
            epsilon = randn(num_tot,1);
            theta_param = mix_Mu{c,1} + ((mix_T{c,1}')\epsilon); %generate global and random effects
            [log_posterior,grad_log_posterior] = obtain_grad_logposterior_logit_some(y,x,theta_param(num_randeffect+1:num_randeffect+num_param,1),theta_param(1:num_randeffect,1),...
                prior,num_period,group,num_some); %computing log h and grad of h
            [log_q,grad_log_q] = obtain_grad_q(theta_param,num_tot,mix_Weights,mix_Mu,mix_T); %compute log q and grad of log q
            grad_mu = (grad_log_posterior - grad_log_q); %grad of \mu
            grad_mu_store(s,:) = grad_mu;
            grad_T = -((mix_T{c,1}')\epsilon)*((mix_T{c,1})\grad_mu)'; %grad of cholesky factor L
            temp_vec=grad_T(sub2ind(size(grad_T),[indx(:,1)'],[indx(:,2)']));
            temp_vec_diag=grad_T(sub2ind(size(grad_T),[indx_diag(:,1)'],[indx_diag(:,2)'])).*mix_T{c,1}(sub2ind(size(mix_T{c,1}),[indx_diag(:,1)'],[indx_diag(:,2)']));
            temp_mat=zeros(num_tot);
            temp_mat(sub2ind(size(temp_mat),indx(:,1),indx(:,2)))=temp_vec';
            temp_mat(sub2ind(size(temp_mat),indx_diag(:,1),indx_diag(:,2)))=temp_vec_diag';
            grad_T = temp_mat(sub2ind(size(temp_mat),indx(:,1),indx(:,2)));
            grad_T_store(s,:)= grad_T;
            LB(s,1) = log_posterior - log_q; 
            
    end
       
    grad_mean_mu = (mean(grad_mu_store,1))';
    grad_mean_T = (mean(grad_T_store,1))';
    
    grad_mean_mu_bar=grad_mean_mu;
    grad_mean_T_bar = grad_mean_T;
    
    iter=1;
    LB_est(iter,1) = mean(LB);
    LB_smooth(1) = LB_est(iter,1);
    t_smooth=1;
    stop=false;
    patience_parameter=50;
    max_best=LB_smooth(1);
    patience=0;
    tic
    for iter = 1:max_iter
        %optimisation
        parfor s=1:num_samples
            
            epsilon = randn(num_tot,1);
            theta_param = mix_Mu{c,1} + ((mix_T{c,1}')\epsilon);
            [log_posterior,grad_log_posterior] = obtain_grad_logposterior_logit_some(y,x,theta_param(num_randeffect+1:num_randeffect+num_param,1),theta_param(1:num_randeffect,1),...
                prior,num_period,group,num_some);
            [log_q,grad_log_q] = obtain_grad_q(theta_param,num_tot,mix_Weights,mix_Mu,mix_T);
            grad_mu = (grad_log_posterior - grad_log_q);
            grad_mu_store(s,:) = grad_mu;
            grad_T = -((mix_T{c,1}')\epsilon)*((mix_T{c,1})\grad_mu)';
            temp_vec=grad_T(sub2ind(size(grad_T),[indx(:,1)'],[indx(:,2)']));
            temp_vec_diag=grad_T(sub2ind(size(grad_T),[indx_diag(:,1)'],[indx_diag(:,2)'])).*mix_T{c,1}(sub2ind(size(mix_T{c,1}),[indx_diag(:,1)'],[indx_diag(:,2)']));
            temp_mat=zeros(num_tot);
            temp_mat(sub2ind(size(temp_mat),indx(:,1),indx(:,2)))=temp_vec';
            temp_mat(sub2ind(size(temp_mat),indx_diag(:,1),indx_diag(:,2)))=temp_vec_diag';
            grad_T = temp_mat(sub2ind(size(temp_mat),indx(:,1),indx(:,2)));
            grad_T_store(s,:)= grad_T;
            LB(s,1) = log_posterior - log_q; 
            
        end
        
        LB_est(iter,1) = mean(LB);
    
        grad_mean_mu = (mean(grad_mu_store,1))';
        grad_mean_T = (mean(grad_T_store,1))';

        grad_mean_mu_bar = adapt_grad_weight_mu*grad_mean_mu_bar+(1-adapt_grad_weight_mu)*grad_mean_mu; 
        grad_mean_T_bar = adapt_grad_weight_T*grad_mean_T_bar+(1-adapt_grad_weight_T)*grad_mean_T; 

        %updating the variational parameters of the first component
        
        m_t_mu{c,1}=adapt_tau_1*m_t_mu{c,1}+(1-adapt_tau_1)*grad_mean_mu_bar;
        v_t_mu{c,1}=adapt_tau_2*v_t_mu{c,1}+(1-adapt_tau_2)*(grad_mean_mu_bar.^2);
        mt_hat_mu{c,1}=m_t_mu{c,1}./(1-(adapt_tau_1.^iter));
        vt_hat_mu{c,1}=v_t_mu{c,1}./(1-(adapt_tau_2.^iter));

        temp_mu = mix_Mu{c,1}+adapt_alpha_mu*(mt_hat_mu{c,1}./(sqrt(vt_hat_mu{c,1})+adapt_epsilon));
        if sum(isnan(temp_mu))>0 | sum(isinf(temp_mu))>0
           mix_Mu{c,1}=mix_Mu{c,1}; 
        else
           mix_Mu{c,1}=temp_mu; 
        end

        m_t_T{c,1}=adapt_tau_1*m_t_T{c,1}+(1-adapt_tau_1)*grad_mean_T_bar;
        v_t_T{c,1}=adapt_tau_2*v_t_T{c,1}+(1-adapt_tau_2)*(grad_mean_T_bar.^2);
        mt_hat_T{c,1}=m_t_T{c,1}./(1-(adapt_tau_1.^iter));
        vt_hat_T{c,1}=v_t_T{c,1}./(1-(adapt_tau_2.^iter));

        delta_T_temp = adapt_alpha_T*(mt_hat_T{c,1}./(sqrt(vt_hat_T{c,1})+adapt_epsilon));
        delta_T(sub2ind(size(delta_T),indx(:,1),indx(:,2)))=delta_T_temp';
        
        if sum(isnan(delta_T))>0 | sum(isinf(delta_T))>0
           mix_T_das{c,1}=mix_T_das{c,1};
           mix_T{c,1}=mix_T{c,1};

        else
           mix_T_das{c,1}=mix_T_das{c,1}+delta_T;
           mix_T{c,1}=mix_T_das{c,1};
           temp11_vec=exp(mix_T_das{c,1}(sub2ind(size(mix_T_das{c,1}),[indx_diag(:,1)'],[indx_diag(:,2)'])));
           mix_T{c,1}(sub2ind(size(mix_T{c,1}),[indx_diag(:,1)'],[indx_diag(:,2)']))=temp11_vec';
        end
        
        if iter>window
           t_smooth = t_smooth + 1;
           LB_smooth(t_smooth) = mean(LB_est(iter-window:iter)); 
       
        end
     
        if iter>window
            if ((LB_smooth(t_smooth)< max_best) ||(abs(LB_smooth(t_smooth)-LB_smooth(t_smooth - 1))<0.00001)) 
                patience = patience + 1;
            else
                patience = 0;
                max_best = LB_smooth(t_smooth);

            end 

        end
        
        mix_T_first(iter,c) = mix_T{c,1}(1,1);
        mix_T_second(iter,c) = mix_T{c,1}(2,2);
        
    end
    cpu_time(c,1) = toc;
    
    else
        %boosting iterations k=2,3,...,K.
        [mix_Mu{c,1}(1:num_randeffect,1),init_T] = initialisation_global_update_used(num_randeffect,num_param,mix_Mu,mix_T,mix_Weights,indx,indx_diag,...
            c,prior,num_period,group,y,x,num_some); %initialise the mean and cholesky factor of random effects.
        mix_Mu{c,1}(num_randeffect+1:num_randeffect+num_param,1) = initialisation_global_parameter(num_randeffect,num_param,mix_Mu(1:c-1),mix_T(1:c-1),mix_Weights,...
            indx,indx_diag,num_chosen,c,prior,num_period,group,y,x,num_some);%initialise global parameters
        
        for j=1:num_randeffect
            mix_T{c,1}(j,j)=init_T(j,1);        
            mix_T_das{c,1}(j,j) = log(init_T(j,1));
        end
        mix_T{c,1}(num_randeffect+1:num_randeffect+num_param,num_randeffect+1:num_randeffect+num_param) = (10*speye(num_param));
        mix_T_das{c,1}(num_randeffect+1:num_randeffect+num_param,num_randeffect+1:num_randeffect+num_param) = log(10)*speye(num_param);
        
        mix_T{c,1} = sparse(mix_T{c,1});
        mix_T_das{c,1} = sparse(mix_T_das{c,1});
        
        %initial values for the weights.
        phi_weight = 0.5;
        if c>2
           mix_Weights = [mix_Weights(1,1:c-2),mix_Weights(1,c-1)*(1-phi_weight),mix_Weights(1,c-1)*phi_weight]; 
        else
           mix_Weights = (1/2)*ones(1,2); 
        end
        tlam_mixw = log(mix_Weights(1,1:end-1))-log(mix_Weights(1,end));
        log_mix_Weights=log(mix_Weights);
                                
        m_t_mu{c,1}=0;
        v_t_mu{c,1}=0;

        m_t_T{c,1}=0;
        v_t_T{c,1}=0;

        m_t_weight = 0;
        v_t_weight = 0;
        
        mt_hat_weight = 0;
        vt_hat_weight = 0;
        
        mt_hat_T{c,1} = 0;
        vt_hat_T{c,1} = 0;
        
        mt_hat_mu{c,1} = 0;
        vt_hat_mu{c,1} = 0;
        %computing the gradient of mu
        [thetaSampled1,logSampDensPerComp1]=SampleFromMixture_Weight(log_mix_Weights,mix_Mu(1:c),mix_T(1:c),num_samples);
        [logRBindicator1,logTotalSampDens1]=CombineMixtureComponents(log_mix_Weights,logSampDensPerComp1); 
        sa = zeros(num_tot,1);
        log_sxxWeights = log_mix_Weights';
        parfor s=1:num_samples
            [lpDens1(s,1),grad1(:,s)] = obtain_grad_logposterior_logit_some(y,x,thetaSampled1(num_randeffect+1:num_randeffect+num_param,s),thetaSampled1(1:num_randeffect,s),...
                    prior,num_period,group,num_some);  
            [log_q(s,1),grad_log_q(:,s)] = obtain_grad_q(thetaSampled1(:,s),num_tot,mix_Weights,mix_Mu,mix_T);   
            lwt = logRBindicator1(c,s) - log_sxxWeights(c);
            sa = sa + weightProd(lwt,grad1(:,s)-grad_log_q(:,s));
        end
        grad_mean_mu_temp = sa./num_samples;
        scalar_temp=lpDens1-logTotalSampDens1';
        temp_temp_temp_mu = ((mix_T{c,1})\(grad_mean_mu_temp));
        grad_mean_mu = ((mix_T{c,1}')\temp_temp_temp_mu);
        
        grad_mean_mu_bar = grad_mean_mu;
        %computing the gradient of cholesky factor
        [gra_log_q_lambda_T,g_lambda_T]=obtain_grad_without_c_OneAtTime_General_Weight(thetaSampled1,mix_Mu,mix_T,num_tot,logRBindicator1,scalar_temp, num_samples,c,indx,indx_diag);
        
        cov_lambda_T=(1/(num_samples-1))*sum((g_lambda_T-mean(g_lambda_T)).*(gra_log_q_lambda_T-mean(gra_log_q_lambda_T)));
        var_lambda_T=(1/(num_samples-1))*sum((gra_log_q_lambda_T-mean(gra_log_q_lambda_T)).^2);
        c_lambda_T=cov_lambda_T./var_lambda_T;
            
        grad_mean_T = mean(g_lambda_T)';
        grad_mean_T_bar = grad_mean_T;
        temp_temp=(lpDens1-log_q);


        %computing the gradients of weights
        if sum(temp_temp>0)==num_samples | sum(temp_temp<0)==num_samples
           [log_sxyWeights,sig]= logMatrixProd(logRBindicator1,(lpDens1-logTotalSampDens1')/num_samples);
           g_m_w2 = exp(log_sxyWeights  - log_sxxWeights) .* sig;
           g_m_w2(sig==0) = 0;
           g_m_w2=g_m_w2';
        else
           g_m_w2=mean(exp(logRBindicator1-log_sxxWeights)'.*(lpDens1-logTotalSampDens1'));
        end
        
        g_m_w = g_m_w2(1,1:c-1) - g_m_w2(1,c); 
        grad_Weights=g_m_w;
        grad_Weights_bar = grad_Weights;
               
        iter=1;
        LB_est=[];
        LB_smooth=[];
        LB_est(iter,1) = mean(scalar_temp);
        LB_smooth(1) = LB_est(iter,1);
        iter=iter+1;
        t_smooth=1;
        stop=false;
        max_best=LB_smooth(1);
        patience=0;
        tic
        %stochastic gradient optimisation
        for iter=1:max_iter
            max_tlam_mixw=max(max(tlam_mixw),0);
            norm_log_mixw=log1p(sum(exp(tlam_mixw - max_tlam_mixw))-1+exp(-max_tlam_mixw))+max_tlam_mixw;
            log_mix_Weights(1,1:c-1)=tlam_mixw-norm_log_mixw; 
            log_mix_Weights(1,c)=-norm_log_mixw;
            mix_Weights=exp(log_mix_Weights);
            %computing the gradient of \mu
            [thetaSampled1,logSampDensPerComp1]=SampleFromMixture_Weight(log_mix_Weights,mix_Mu(1:c),mix_T(1:c),num_samples);
            [logRBindicator1,logTotalSampDens1]=CombineMixtureComponents(log_mix_Weights,logSampDensPerComp1); 
            sa = zeros(num_tot,1);
            log_sxxWeights = log_mix_Weights';
            parfor s=1:num_samples
                [lpDens1(s,1),grad1(:,s)] = obtain_grad_logposterior_logit_some(y,x,thetaSampled1(num_randeffect+1:num_randeffect+num_param,s),thetaSampled1(1:num_randeffect,s),...
                    prior,num_period,group,num_some); 
                [log_q(s,1),grad_log_q(:,s)] = obtain_grad_q(thetaSampled1(:,s),num_tot,mix_Weights,mix_Mu,mix_T);
                lwt = logRBindicator1(c,s) - log_sxxWeights(c);
                sa = sa + weightProd(lwt,grad1(:,s)-grad_log_q(:,s));
            end
            grad_mean_mu_temp = sa./num_samples;
            temp_temp_temp_mu = ((mix_T{c,1})\(grad_mean_mu_temp));
            grad_mean_mu = ((mix_T{c,1}')\temp_temp_temp_mu);
            temp_temp=(lpDens1-log_q);
            LB_est(iter,1) = mean(temp_temp);
            %computing the gradients of weights
            if sum(temp_temp>0)==num_samples | sum(temp_temp<0)==num_samples
               [log_sxyWeights,sig]= logMatrixProd(logRBindicator1,(lpDens1-logTotalSampDens1')/num_samples);
               g_m_w2 = exp(log_sxyWeights  - log_sxxWeights) .* sig;
               g_m_w2(sig==0) = 0;
               g_m_w2=g_m_w2';
            else
               g_m_w2=mean(exp(logRBindicator1-log_sxxWeights)'.*(lpDens1-logTotalSampDens1'));
            end
            
            %computing the gradient of cholesky factor
            [thetaSampled2,logSampDensPerComp2]=SampleFromMixture_Weight(log_mix_Weights,mix_Mu(1:c),mix_T(1:c),num_samples);
            [logRBindicator2,logTotalSampDens2]=CombineMixtureComponents(log_mix_Weights,logSampDensPerComp2); 
            parfor s=1:num_samples
                [lpDens2(s,1),grad2(:,s)] = obtain_grad_logposterior_logit_some(y,x,thetaSampled2(num_randeffect+1:num_randeffect+num_param,s),thetaSampled2(1:num_randeffect,s),...
                    prior,num_period,group,num_some); 
                [log_q(s,1),grad_log_q(:,s)] = obtain_grad_q(thetaSampled2(:,s),num_tot,mix_Weights,mix_Mu,mix_T);
                
            end
            
            scalar_temp = (lpDens2-log_q);
            [gra_log_q_lambda_T,g_lambda_T_for_c,g_lambda_T]=obtain_grad_with_c_OneAtTimeGeneral_Weight(thetaSampled2,...
                mix_Mu,mix_T,num_tot,logRBindicator2,scalar_temp,num_samples,c,c_lambda_T,indx,indx_diag);

            cov_lambda_T=(1/(num_samples-1))*sum((g_lambda_T_for_c-mean(g_lambda_T_for_c)).*(gra_log_q_lambda_T-mean(gra_log_q_lambda_T)));
            var_lambda_T=(1/(num_samples-1))*sum((gra_log_q_lambda_T-mean(gra_log_q_lambda_T)).^2);
            c_lambda_T=cov_lambda_T./var_lambda_T;
            
            grad_mean_T = mean(g_lambda_T)';
            
            
            g_m_w = g_m_w2(1,1:c-1) - g_m_w2(1,c); 
            grad_Weights=g_m_w;
            
            grad_mean_mu_bar = adapt_grad_weight_mu*grad_mean_mu_bar+(1-adapt_grad_weight_mu)*grad_mean_mu; 
            grad_mean_T_bar = adapt_grad_weight_T*grad_mean_T_bar+(1-adapt_grad_weight_T)*grad_mean_T; 
            grad_Weights_bar = adapt_grad_weight_weight*grad_Weights_bar+(1-adapt_grad_weight_weight)*grad_Weights; 

            
            %updating the weight
            m_t_weight=adapt_tau_1*m_t_weight+(1-adapt_tau_1)*grad_Weights_bar;
            v_t_weight=adapt_tau_2*v_t_weight+(1-adapt_tau_2)*grad_Weights_bar.^2;
            mt_hat_weight=m_t_weight./(1-(adapt_tau_1.^iter));
            vt_hat_weight=v_t_weight./(1-(adapt_tau_2.^iter));
            temp_weight=tlam_mixw+adapt_alpha_weight*(mt_hat_weight./(sqrt(vt_hat_weight)+adapt_epsilon));
            
            
            
            
            if sum(isnan(temp_weight))>0 | sum(isinf(temp_weight))>0
                tlam_mixw = tlam_mixw;
            else
                tlam_mixw = temp_weight;
            end
            
            m_t_T{c,1}=adapt_tau_1*m_t_T{c,1}+(1-adapt_tau_1)*grad_mean_T_bar;
            v_t_T{c,1}=adapt_tau_2*v_t_T{c,1}+(1-adapt_tau_2)*(grad_mean_T_bar.^2);
            mt_hat_T{c,1}=m_t_T{c,1}./(1-(adapt_tau_1.^iter));
            vt_hat_T{c,1}=v_t_T{c,1}./(1-(adapt_tau_2.^iter));
 
            delta_T_temp = adapt_alpha_T*(mt_hat_T{c,1}./(sqrt(vt_hat_T{c,1})+adapt_epsilon));
            delta_T(sub2ind(size(delta_T),indx(:,1),indx(:,2)))=delta_T_temp';
            
            if sum(isnan(delta_T_temp))>0 | sum(isinf(delta_T_temp))>0
               mix_T_das{c,1}=mix_T_das{c,1};
               mix_T{c,1}=mix_T{c,1}; 
            
            else
               mix_T_das{c,1}=mix_T_das{c,1}+delta_T;
               mix_T{c,1}=mix_T_das{c,1}; 
               temp11_vec=exp(mix_T_das{c,1}(sub2ind(size(mix_T_das{c,1}),[indx_diag(:,1)'],[indx_diag(:,2)'])));
               mix_T{c,1}(sub2ind(size(mix_T{c,1}),[indx_diag(:,1)'],[indx_diag(:,2)']))=temp11_vec';
            
            end   

            
            m_t_mu{c,1}=adapt_tau_1*m_t_mu{c,1}+(1-adapt_tau_1)*grad_mean_mu_bar;
            v_t_mu{c,1}=adapt_tau_2*v_t_mu{c,1}+(1-adapt_tau_2)*(grad_mean_mu_bar.^2);
            mt_hat_mu{c,1}=m_t_mu{c,1}./(1-(adapt_tau_1.^iter));
            vt_hat_mu{c,1}=v_t_mu{c,1}./(1-(adapt_tau_2.^iter)); 
            temp_mu = mix_Mu{c,1}+(adapt_alpha_mu*(mt_hat_mu{c,1}./(sqrt(vt_hat_mu{c,1})+adapt_epsilon)));
            if sum(isnan(temp_mu))>0 | sum(isinf(temp_mu))>0
               mix_Mu{c,1}=mix_Mu{c,1}; 
            else
               mix_Mu{c,1}=temp_mu; 
            end
            
            if iter>window
                t_smooth = t_smooth + 1;
                LB_smooth(t_smooth) = mean(LB_est(iter-window:iter)); 
       
            end
    
            if iter>window
                if ((LB_smooth(t_smooth)< max_best) ||(abs(LB_smooth(t_smooth)-LB_smooth(t_smooth - 1))<0.00001)) 
                    patience = patience + 1;
                else
                    patience = 0;
                    max_best = LB_smooth(t_smooth);

                end 

            end
            
            
            mix_T_first(iter,c) = mix_T{c,1}(1,1);
            mix_T_second(iter,c) = mix_T{c,1}(2,2);
            
            
        end
        cpu_time(c,1) = toc;
        
        max_tlam_mixw=max(max(tlam_mixw),0);
        norm_log_mixw=log1p(sum(exp(tlam_mixw - max_tlam_mixw))-1+exp(-max_tlam_mixw))+max_tlam_mixw;
        log_mix_Weights(1,1:c-1)=tlam_mixw-norm_log_mixw; 
        log_mix_Weights(1,c)=-norm_log_mixw;
        mix_Weights=exp(log_mix_Weights); 

        



    end
    
    
    %computing the Stilde measure. 
    [measure_latent,MSE_latent] = calculating_RKL(num_randeffect,num_param,mix_Mu,mix_T,mix_Weights,prior,num_period,group,y,x,num_some);


    
    
    if c<nrComponent
       chosen_index = randsample(c,1,true, mix_Weights); 
       temp_mix_mu = mix_Mu{chosen_index,1};
       temp_mix_T = mix_T{chosen_index,1};
       temp_mix_T_das = mix_T_das{chosen_index,1};
       temp_mix_w = mix_Weights(1,chosen_index);
       mix_Mu{chosen_index,1} = mix_Mu{c,1};
       mix_T{chosen_index,1} = mix_T{c,1};
       mix_T_das{chosen_index,1} = mix_T_das{c,1};
       mix_Weights(1,chosen_index) = mix_Weights(1,c);
       mix_Mu{c,1} = temp_mix_mu;
       mix_T{c,1} = temp_mix_T;
       mix_T_das{c,1} = temp_mix_T_das;
       mix_Weights(1,c) = temp_mix_w;

       
    end
    
    mix_Weights_old=mix_Weights_actual;
    mod_name=['mixCop_Global',num2str(c),'_',num2str(num_some),'_CV_Z.mat'];
     save(mod_name,'mix_Weights','mix_Mu','mix_T','mix_T_das','LB_est','mix_Weights_old','measure_latent','LB_smooth','mix_T_first','mix_T_second','MSE_latent','cpu_time');

    
end


%             for k=1:c
%                 for s=1:num_samples
%                     epsilon = randn(num_param,1);
%                     theta_param = mix_Mu{k,1} + ((mix_T{k,1}')\epsilon);
%                     [log_posterior(s,1),grad_log_posterior] = obtain_grad_param_mixNormal(theta_param,num_param,weight_mix_true,mu_true,precs_true);
%                     [log_q(s,1),grad_log_q] = obtain_grad_q(theta_param,num_param,mix_Weights,mix_Mu,mix_T);
%                     grad_mu_ind(s,:) = grad_log_posterior - grad_log_q;
%                 
%                     grad_T = -((mix_T{k,1}')\epsilon)*((mix_T{k,1})\grad_mu_ind(s,:)')';
%                     temp_vec=grad_T(sub2ind(size(grad_T),[indx(:,1)'],[indx(:,2)']));
%                     temp_vec_diag=grad_T(sub2ind(size(grad_T),[indx_diag(:,1)'],[indx_diag(:,2)'])).*mix_T{k,1}(sub2ind(size(mix_T{k,1}),[indx_diag(:,1)'],[indx_diag(:,2)']));
%                     temp_mat=zeros(num_param);
%                     temp_mat(sub2ind(size(temp_mat),indx(:,1),indx(:,2)))=temp_vec';
%                     temp_mat(sub2ind(size(temp_mat),indx_diag(:,1),indx_diag(:,2)))=temp_vec_diag';
%                     grad_T = temp_mat(sub2ind(size(temp_mat),indx(:,1),indx(:,2)));
%                     grad_T_ind(s,:)= grad_T;
%                     
%                 end
%                                  
%                 
%  
%                 grad_T_sum = grad_T_sum + mix_Weights_old(k)*sum(weight.*grad_T_ind); 
%                 grad_mu_sum = grad_mu_sum + mix_Weights_old(k)*sum(weight.*grad_mu_ind);
%             end
            
            %FKL_store(iter,1) = FKL;
            %FKL



%[log_q_Kmin(s,1),~] = obtain_grad_q(theta_param,num_param,mix_Weights_old,mix_Mu(1:c-1,1),mix_T(1:c-1,1));
%grad_weight_ind_num = (normpdf_precs(theta_param, num_param, mix_Mu{c,1}, mix_T{c,1}) - exp(log_q_Kmin(s,1)));
%grad_weight_ind_den = (1 - phi_weight)*exp(log_q_Kmin(s,1)) + phi_weight*normpdf_precs(theta_param, num_param, mix_Mu{c,1}, mix_T{c,1});
%grad_weight_ind(s,1) = -(grad_weight_ind_num/grad_weight_ind_den);

%epsilon=randn(num_param,num_samples);
%options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton');
%[lambda, fval]=fminunc(@(lambda) compute_lower_bound_SAA(y,x,lambda,epsilon,num_param,indx,indx_diag),lambda0,options);

%for i=1:n_iter
    
% temp_mat = zeros(num_param);
% A = randn(5,5);
% temp_A = temp_mat(sub2ind(size(A),[indx(:,1)'],[indx(:,2)']));
% temp_A = A(sub2ind(size(A),[indx(:,1)'],[indx(:,2)']));
% temp_mat(sub2ind(size(temp_mat),[indx(:,1)'],[indx(:,2)'])) = temp_A';

    
%[x,fval] = fminunc(@(x) myfun(x,y),x0);

      

    
%     beta=mu+B*z+d.*epsilon;    
%     grad_log_posterior=obtain_grad_logposterior_logit(y,x,beta);
%     grad_mu=grad_log_posterior;
%    
%     %grad_B=grad_log_posterior*z'+((B*B'+diag([d.^2]))\(B*z+d.*epsilon))*z';
%     temp_wood=compute_woodbury(B,d);
%     grad_B=grad_log_posterior*z'+(temp_wood*(B*z+d.*epsilon))*z';
%     %grad_B(1,2)=0;
%     
%     grad_d=diag(grad_log_posterior*epsilon'+(temp_wood*(B*z+d.*epsilon))*epsilon');
%     
%     E_g2_mu=adapt_rho*E_g2_mu+(1-adapt_rho).*(grad_mu.^2);
%     delta_mu=(sqrt(E_delta2_mu+adapt_eps)./sqrt(E_g2_mu+adapt_eps)).*grad_mu;
%     E_delta2_mu=adapt_rho*E_delta2_mu+(1-adapt_rho).*(delta_mu.^2);
%     mu=mu+delta_mu;
%     
%     E_g2_B=adapt_rho*E_g2_B+(1-adapt_rho).*(grad_B.^2);
%     delta_B=(sqrt(E_delta2_B+adapt_eps)./sqrt(E_g2_B+adapt_eps)).*grad_B;
%     E_delta2_B=adapt_rho*E_delta2_B+(1-adapt_rho).*(delta_B.^2);
%     B=B+delta_B;
%     
%     E_g2_d=adapt_rho*E_g2_d+(1-adapt_rho).*(grad_d.^2);
%     delta_d=(sqrt(E_delta2_d+adapt_eps)./sqrt(E_g2_d+adapt_eps)).*grad_d;
%     E_delta2_d=adapt_rho*E_delta2_d+(1-adapt_rho).*(delta_d.^2);
%     d=d+delta_d;
%     
%     [lower_bound(i,1)]=compute_lower_bound(y,x,beta,B,d,z,epsilon);
    
%end
%save('logistic_grad.mat','mu','B','d','lower_bound');
