%This code implements Local boosting for logistic regression model with Z
%distribution for the prior for random effects. 

parpool(10)
load('polypharm_data.mat'); %load the dataset
num_param=size(data(:,3:end),2)+1; %number of global parameters
num_randeffect = length(unique(data(:,2))); % number of random effects
num_tot = num_randeffect+num_param;
num_period = length(data(:,1))/num_randeffect;

mix_Mu{1,1}=zeros(num_tot,1); %initial values for mean of the first component
mix_T{1,1}=log(100)*speye(num_tot); %initial values for the cholesky factor of the first component
mix_T_das{1,1}=log(100)*speye(num_tot);
indx = obtain_index_full_randeffect(num_randeffect,num_param);% finding index of the random effects
indx_diag(:,1) = (1:num_tot)';
indx_diag(:,2) = (1:num_tot)';

prob = 0.1;
move = 1;
delta_T=speye(num_tot);
num_samples = 10; %number of samples
prior.sig2_beta = 1;
prior.sig2_alpha = [0.0001,1];
prior.mu = [0,0];
prior.w_a = [0.5,0.5];
prior.df = 3;
y = data(:,1); %response variable
group = data(:,2);

n = length(data(:,1));
x = [ones(n,1),data(:,3:end)]; %matrix of predictors

nrComponent = 20; %number of components
num_chosen = 50; % number of latent variables improved in each boosting iteration. 
adapt_tau_1 = 0.9; %ADAM hyperparameters
adapt_tau_2 = 0.99;
adapt_epsilon = 10^-8;
adapt_alpha_mu = 0.01;
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


max_iter = 5000; %number of iterations
mix_Weights = 1;

id_mean_update_store{1,1}=[];
measure_latent = [];
adapt_grad_weight_mu = 0;
adapt_grad_weight_T = 0;
adapt_grad_weight_weight = 0;

num_some = 50; %the number of random effects with complex prior distributions
window = 100;
for c=1:nrComponent
    c
    if c==1
    %optimisation for the first component using stochastic gradient ascent
    %with reparameterisation trick
    parfor s=1:num_samples
            
            epsilon = randn(num_tot,1);
            theta_param = mix_Mu{c,1} + ((mix_T{c,1}')\epsilon); %generate global and random effects
            [log_posterior,grad_log_posterior] = obtain_grad_logposterior_logit_some(y,x,theta_param(num_randeffect+1:num_randeffect+num_param,1),theta_param(1:num_randeffect,1),...
                prior,num_period,group,num_some); %computing log h and grad of h
            [log_q,grad_log_q] = obtain_grad_q(theta_param,num_tot,mix_Weights,mix_Mu,mix_T);
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
        if (patience>patience_parameter) || (iter>max_iter)
           stop=true;
        end
    
           
        mix_T_first(iter,c) = mix_T{c,1}(1,1);
        mix_T_second(iter,c) = mix_T{c,1}(2,2);
        
    end
    cpu_time(c,1) = toc;
    else
        
        %boosting iterations k=2,3,...,K.
        %initialisation variational parameters. 
        mix_Mu{c,1} = mix_Mu{c-1,1};
        mix_T{c,1} = mix_T{c-1,1};
        mix_T_das{c,1} = mix_T_das{c-1,1};

        phi_weight = 0.5;
        split_weight = [mix_Weights(1,c-1)*(1-phi_weight), mix_Weights(1,c-1)*phi_weight];
        if c>2
           mix_Weights = [mix_Weights(1,1:c-2),split_weight(1,1),split_weight(1,2)]; 
           
        else
           mix_Weights = (1/2)*ones(1,2); 
        end
        tlam_mixw = log(split_weight(1,1))-log(split_weight(1,2));
        log_mix_Weights=log(mix_Weights);
        
        id_T_update = [];
        if move == 2
            id_mean_update_store{c,1} = id_mean_update;
        end
        for l = 1:length(id_mean_update)
            id_T_update = [id_T_update;find(indx(:,2) == id_mean_update(l,1))];
        end
        id_T_update_vec = unique(id_T_update);
        id_T_update_vec_sort = sort(id_T_update_vec);
        
        index_mean_update = zeros(num_tot,1);
        index_T_update = zeros(length(indx),1);
        
        index_mean_update(id_mean_update,1) = 1;
        index_T_update(id_T_update_vec_sort,1) = 1; 
        
        
        
        if move == 2
            
            for j=1:length(id_mean_update)
                mix_Mu{c,1}(id_mean_update(j),1) = init_mu(id_mean_update(j),1);        
            end
            
            indx_below = sum(id_T_update_vec_sort<=num_randeffect);
            indx_upper = length(id_T_update_vec_sort) - indx_below;
            temp_temp_T_das = (mix_T_das{c,1}(sub2ind(size(mix_T_das{c,1}),[indx(:,1)'],[indx(:,2)'])))';
            
            for j=1:indx_below
                temp_temp_T_das(id_T_update_vec_sort(j,1)) = log(init_T(id_T_update_vec_sort(j,1)));
            end
            temp_temp_T_das(id_T_update_vec_sort(indx_below+1:length(id_T_update_vec_sort),1),1) = [0.001*ones(num_param*length(id_mean_update),1)]';
            mix_T_das{c,1}(sub2ind(size(mix_T_das{c,1}),[indx(:,1)'],[indx(:,2)']))= temp_temp_T_das';

            
            temp_temp_T = (mix_T{c,1}(sub2ind(size(mix_T{c,1}),[indx(:,1)'],[indx(:,2)'])))';
            for j=1:indx_below
                temp_temp_T(id_T_update_vec_sort(j,1)) = (init_T(id_T_update_vec_sort(j,1)));
            end
            temp_temp_T(id_T_update_vec_sort(indx_below+1:length(id_T_update_vec_sort),1),1) = [0.001*ones(num_param*length(id_mean_update),1)]';
            
            mix_T{c,1}(sub2ind(size(mix_T{c,1}),[indx(:,1)'],[indx(:,2)']))= temp_temp_T';
                    
        else
            for j=1:length(id_mean_update)
                mix_Mu{c,1}(id_mean_update(j),1) = init_mu(j,1);        
            end
            
            
            tot_T_update = (num_param*(num_param-1))/2+num_param;
            temp_temp_T_das = (mix_T_das{c,1}(sub2ind(size(mix_T_das{c,1}),[indx(:,1)'],[indx(:,2)'])))';
            temp_temp_T_das(id_T_update_vec,1) = [log(10)*ones(length(id_mean_update),1);0.001*ones(tot_T_update-length(id_mean_update),1)]';            
            mix_T_das{c,1}(sub2ind(size(mix_T_das{c,1}),[indx(:,1)'],[indx(:,2)']))= temp_temp_T_das';
        
            temp_temp_T = (mix_T{c,1}(sub2ind(size(mix_T{c,1}),[indx(:,1)'],[indx(:,2)'])))';
            temp_temp_T(id_T_update_vec,1) = [(10)*ones(length(id_mean_update),1);0.001*ones(tot_T_update-length(id_mean_update),1)]';
            mix_T{c,1}(sub2ind(size(mix_T{c,1}),[indx(:,1)'],[indx(:,2)']))= temp_temp_T';
            
            
        end
        
        
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
        %computing grad mu
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
        
        parfor s=1:num_samples
            
            epsilon = randn(num_tot,1);
            theta_param = mix_Mu{c,1} + ((mix_T{c,1}')\epsilon);
            [log_posterior,grad_log_posterior] = obtain_grad_logposterior_logit_some(y,x,theta_param(num_randeffect+1:num_randeffect+num_param,1),theta_param(1:num_randeffect,1),...
                prior,num_period,group,num_some);
            [log_q,grad_log_q] = obtain_grad_q(theta_param,num_tot,mix_Weights,mix_Mu,mix_T);
            temp_mu = (grad_log_posterior - grad_log_q);
            grad_T = -((mix_T{c,1}')\epsilon)*((mix_T{c,1})\temp_mu)';
            temp_vec=grad_T(sub2ind(size(grad_T),[indx(:,1)'],[indx(:,2)']));
            temp_vec_diag=grad_T(sub2ind(size(grad_T),[indx_diag(:,1)'],[indx_diag(:,2)'])).*mix_T{c,1}(sub2ind(size(mix_T{c,1}),[indx_diag(:,1)'],[indx_diag(:,2)']));
            temp_mat=zeros(num_tot);
            temp_mat(sub2ind(size(temp_mat),indx(:,1),indx(:,2)))=temp_vec';
            temp_mat(sub2ind(size(temp_mat),indx_diag(:,1),indx_diag(:,2)))=temp_vec_diag';
            grad_T = temp_mat(sub2ind(size(temp_mat),indx(:,1),indx(:,2)));
            grad_T_store(s,:)= grad_T;
            LB(s,1) = log_posterior - log_q; 
            
        end
            
        grad_mean_T = (mean(grad_T_store,1))';
        grad_mean_T_bar = grad_mean_T;
        %computing grad weights
        temp_temp=(lpDens1-log_q);
        if sum(temp_temp>0)==num_samples | sum(temp_temp<0)==num_samples
           [log_sxyWeights,sig]= logMatrixProd(logRBindicator1,(lpDens1-logTotalSampDens1')/num_samples);
           g_m_w2 = exp(log_sxyWeights  - log_sxxWeights) .* sig;
           g_m_w2(sig==0) = 0;
           g_m_w2=g_m_w2';
        else
           g_m_w2=mean(exp(logRBindicator1-log_sxxWeights)'.*(lpDens1-logTotalSampDens1'));
        end
        
        g_m_w = g_m_w2(1,c-1) - g_m_w2(1,c); 
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
        %optimisation
        for iter=1:max_iter
            max_tlam_mixw=max(max(tlam_mixw),0);
            norm_log_mixw=log1p(sum(exp(tlam_mixw - max_tlam_mixw))-1+exp(-max_tlam_mixw))+max_tlam_mixw;
            log_mix_Weights(1,c-1)=(tlam_mixw-norm_log_mixw)+log(mix_Weights_old(1,c-1)); 
            log_mix_Weights(1,c)=(-norm_log_mixw)+log(mix_Weights_old(1,c-1));
            mix_Weights=exp(log_mix_Weights);
            
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
            
            if sum(temp_temp>0)==num_samples | sum(temp_temp<0)==num_samples
               [log_sxyWeights,sig]= logMatrixProd(logRBindicator1,(lpDens1-logTotalSampDens1')/num_samples);
               g_m_w2 = exp(log_sxyWeights  - log_sxxWeights) .* sig;
               g_m_w2(sig==0) = 0;
               g_m_w2=g_m_w2';
            else
               g_m_w2=mean(exp(logRBindicator1-log_sxxWeights)'.*(lpDens1-logTotalSampDens1'));
            end
            
            scalar_temp = (lpDens1-log_q);
            
            
            g_m_w = g_m_w2(1,c-1) - g_m_w2(1,c); 
            grad_Weights=g_m_w;
            
            
            grad_mean_mu_bar = adapt_grad_weight_mu*grad_mean_mu_bar+(1-adapt_grad_weight_mu)*grad_mean_mu; 
            grad_Weights_bar = adapt_grad_weight_weight*grad_Weights_bar+(1-adapt_grad_weight_weight)*grad_Weights; 
            
            
            
            m_t_weight=adapt_tau_1*m_t_weight+(1-adapt_tau_1)*grad_Weights_bar;
            v_t_weight=adapt_tau_2*v_t_weight+(1-adapt_tau_2)*grad_Weights_bar.^2;
            mt_hat_weight=m_t_weight./(1-(adapt_tau_1.^iter));
            vt_hat_weight=v_t_weight./(1-(adapt_tau_2.^iter));
            temp_weight=tlam_mixw+(adapt_alpha_weight*(mt_hat_weight./(sqrt(vt_hat_weight)+adapt_epsilon)));
            
            if sum(isnan(temp_weight))>0 | sum(isinf(temp_weight))>0
                tlam_mixw = tlam_mixw;
            else
                tlam_mixw = temp_weight;
            end
            
            

            
            m_t_mu{c,1}=adapt_tau_1*m_t_mu{c,1}+(1-adapt_tau_1)*grad_mean_mu_bar;
            v_t_mu{c,1}=adapt_tau_2*v_t_mu{c,1}+(1-adapt_tau_2)*(grad_mean_mu_bar.^2);
            mt_hat_mu{c,1}=m_t_mu{c,1}./(1-(adapt_tau_1.^iter));
            vt_hat_mu{c,1}=v_t_mu{c,1}./(1-(adapt_tau_2.^iter)); 
            temp_mu = mix_Mu{c,1}+(adapt_alpha_mu*(mt_hat_mu{c,1}./(sqrt(vt_hat_mu{c,1})+adapt_epsilon))).*index_mean_update;
            if sum(isnan(temp_mu))>0 | sum(isinf(temp_mu))>0
               mix_Mu{c,1}=mix_Mu{c,1}; 
            else
               mix_Mu{c,1}=temp_mu; 
            end
            
            parfor s=1:num_samples
            
                epsilon = randn(num_tot,1);
                theta_param = mix_Mu{c,1} + ((mix_T{c,1}')\epsilon);
                [log_posterior,grad_log_posterior] = obtain_grad_logposterior_logit_some(y,x,theta_param(num_randeffect+1:num_randeffect+num_param,1),theta_param(1:num_randeffect,1),...
                    prior,num_period,group,num_some);
                [log_q,grad_log_q] = obtain_grad_q(theta_param,num_tot,mix_Weights,mix_Mu,mix_T);
                temp_mu = (grad_log_posterior - grad_log_q);
                grad_T = -((mix_T{c,1}')\epsilon)*((mix_T{c,1})\temp_mu)';
                temp_vec=grad_T(sub2ind(size(grad_T),[indx(:,1)'],[indx(:,2)']));
                temp_vec_diag=grad_T(sub2ind(size(grad_T),[indx_diag(:,1)'],[indx_diag(:,2)'])).*mix_T{c,1}(sub2ind(size(mix_T{c,1}),[indx_diag(:,1)'],[indx_diag(:,2)']));
                temp_mat=zeros(num_tot);
                temp_mat(sub2ind(size(temp_mat),indx(:,1),indx(:,2)))=temp_vec';
                temp_mat(sub2ind(size(temp_mat),indx_diag(:,1),indx_diag(:,2)))=temp_vec_diag';
                grad_T = temp_mat(sub2ind(size(temp_mat),indx(:,1),indx(:,2)));
                grad_T_store(s,:)= grad_T;
                LB(s,1) = log_posterior - log_q; 
            
            end


            grad_mean_T = mix_Weights(1,c)*(mean(grad_T_store,1))';
            grad_mean_T_bar = adapt_grad_weight_T*grad_mean_T_bar+(1-adapt_grad_weight_T)*grad_mean_T; 
            
            m_t_T{c,1}=adapt_tau_1*m_t_T{c,1}+(1-adapt_tau_1)*grad_mean_T_bar;
            v_t_T{c,1}=adapt_tau_2*v_t_T{c,1}+(1-adapt_tau_2)*(grad_mean_T_bar.^2);
            mt_hat_T{c,1}=m_t_T{c,1}./(1-(adapt_tau_1.^iter));
            vt_hat_T{c,1}=v_t_T{c,1}./(1-(adapt_tau_2.^iter));
 
            delta_T_temp = adapt_alpha_T*(mt_hat_T{c,1}./(sqrt(vt_hat_T{c,1})+adapt_epsilon)).*index_T_update;
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
            
%             
            
       
            
            if iter>window
                t_smooth = t_smooth + 1;
                LB_smooth(t_smooth) = mean(LB_est(iter-window:iter)); 
       
            end
    %stopping rule 
            if iter>window
                if ((LB_smooth(t_smooth)< max_best) ||(abs(LB_smooth(t_smooth)-LB_smooth(t_smooth - 1))<0.00001)) 
                    patience = patience + 1;
                else
                    patience = 0;
                    max_best = LB_smooth(t_smooth);

                end 

            end
            if (patience>patience_parameter) || (iter>max_iter)
               stop=true;
            end

           mix_T_first(iter,c) = mix_T{c,1}(1,1);
        mix_T_second(iter,c) = mix_T{c,1}(2,2);

                   
            
        end
        cpu_time(c,1) = toc;
        
        max_tlam_mixw=max(max(tlam_mixw),0);
        norm_log_mixw=log1p(sum(exp(tlam_mixw - max_tlam_mixw))-1+exp(-max_tlam_mixw))+max_tlam_mixw;
        log_mix_Weights(1,c-1)=(tlam_mixw-norm_log_mixw)+log(mix_Weights_old(1,c-1)); 
        log_mix_Weights(1,c)=(-norm_log_mixw)+log(mix_Weights_old(1,c-1));
        mix_Weights=exp(log_mix_Weights);
        
        



    end
    
    [measure_latent] = calculating_RKL(num_randeffect,num_param,mix_Mu,mix_T,mix_Weights,prior,num_period,group,y,x,num_some);
        
    if c<nrComponent
       chosen_index = find(mix_Weights == max(mix_Weights));
 
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

       if rand<prob
       
           id_mean_update = [num_randeffect+1:num_randeffect+num_param]';
           [init_mu] = initialisation_global_parameter(num_randeffect,num_param,mix_Mu,mix_T,mix_Weights,indx,indx_diag,num_chosen,c,prior,num_period,group,y,x,num_some);
           move = 1;
           
       else
           
           [id_mean_update,init_mu,init_T]=initialisation_local_update_used(num_randeffect,num_param,mix_Mu,mix_T,mix_Weights,indx,indx_diag,num_chosen,c,prior,num_period,group,y,x,num_some);
           move=2;
        end
       
       

    end
    mix_Weights_old=mix_Weights;
    mod_name=['mixCop_Local',num2str(c),'_','some_',num2str(num_chosen),'_reparam_mixT_VS.mat'];
    save(mod_name,'mix_Weights','mix_Mu','mix_T','mix_T_das','LB_est','mix_Weights_old','id_mean_update_store','measure_latent','LB_smooth','mix_T_first','mix_T_second',...
        'cpu_time','prior');

end

        
