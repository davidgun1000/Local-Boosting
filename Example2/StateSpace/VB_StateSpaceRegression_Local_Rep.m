%Factor structure VB
%parpool(10)
%parpool(4)
load('Zdata.mat');
num_param=1;
dim_states=4;
length_data = length(y);
num_states = length(y);
num_tot = dim_states*num_states+num_param;
num_chosen = [];
mix_Mu{1,1}=zeros(num_tot,1);
mix_T{1,1}=100*speye(num_tot);
mix_T_das{1,1}=log(100)*speye(num_tot);
indx_col = chol_pattern_indices(length_data, dim_states, num_param);
indx = [indx_col.I,indx_col.J];
indx_diag(:,1) = (1:num_tot)';
indx_diag(:,2) = (1:num_tot)';


prob = 0;
move = 1;
delta_T=speye(num_tot);
num_samples = 10;
prior.hp_sig2 = 100;
prior.mu = 0;
prior.s = 0.1;
prior.a = 0.5;
prior.b = 0.5;
prior.sig2 = 1;

n = length(y);

nrComponent = 2;
num_chosen = 50;
adapt_tau_1 = 0.9;
adapt_tau_2 = 0.99;
adapt_epsilon = 10^-8;
adapt_alpha_mu = 0.01;
adapt_alpha_T = 0.001;
adapt_alpha_T2 = 0.0005;
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


max_iter = 5000;
mix_Weights = 1;

id_mean_update_store{1,1}=[];
measure_latent = [];
adapt_grad_weight_mu = 0;
adapt_grad_weight_T = 0;
adapt_grad_weight_weight = 0;

num_some = 50;
window = 100;
for c=1:nrComponent
    c
    if c==1

    for s=1:num_samples
            
            epsilon = randn(num_tot,1);
            theta_param = mix_Mu{c,1} + ((mix_T{c,1}')\epsilon);
            [log_posterior,grad_log_posterior] = obtain_grad_param_statespace_sv_Z_higherdim(theta_param(dim_states*num_states+1:dim_states*num_states+num_param,1),theta_param(1:dim_states*num_states,1),y,num_param,...
                num_states,prior,num_some,dim_states,[x1,x2,x3]);
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
    
    %while ~stop
    for iter = 1:max_iter
        %iter
        %tic
        %mix_Mu{c,1}(end,1)
        parfor s=1:num_samples
            
            epsilon = randn(num_tot,1);
            theta_param = mix_Mu{c,1} + ((mix_T{c,1}')\epsilon);
            [log_posterior,grad_log_posterior] = obtain_grad_param_statespace_sv_Z_higherdim(theta_param(dim_states*num_states+1:dim_states*num_states+num_param,1),theta_param(1:dim_states*num_states,1),y,num_param,...
                num_states,prior,num_some,dim_states,[x1,x2,x3]);
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
        %toc
        
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
    
        
        
    end

    else
        
      
       mix_Mu{c,1} = mix_Mu{c-1,1};
       mix_T{c,1} = mix_T{c-1,1};
       mix_T_das{c,1} = mix_T_das{c-1,1};

       phi_weight = 0.5;
       split_weight = [mix_Weights(1,c-1)*(1-phi_weight), mix_Weights(1,c-1)*phi_weight];
       if c>2
          mix_Weights = [mix_Weights(1,1:c-2),split_weight(1,1),split_weight(1,2)]; 
           
       else
          mix_Weights = (1/2)*ones(1,2); 
           %mix_Weights = [0.8,0.2];
       end
       tlam_mixw = log(split_weight(1,1))-log(split_weight(1,2));
       log_mix_Weights=log(mix_Weights);
        
       id_T_update = [];
       id_mean_update_store{c,:} = [id_mean_update_latent];
        
        for l = 1:length(id_mean_update_latent)
            if id_mean_update_latent(l) == num_states
               id_T_update = [id_T_update;indx_col.par.Lii{id_mean_update_latent(l)}';indx_col.par.LGi{id_mean_update_latent(l)}'];
            else
               id_T_update = [id_T_update;indx_col.par.Lii{id_mean_update_latent(l)}';indx_col.par.LGi{id_mean_update_latent(l)}';indx_col.par.Ltilde{id_mean_update_latent(l)}'];
            end
        end
        %id_T_update = [id_T_update;indx_col.par.LG];
%         for l = 1:length(id_mean_update)
%             id_T_update = [id_T_update;find(indx(:,2) == id_mean_update(l,1))];
%         end
%         id_T_update_vec = unique(id_T_update);
        
        index_mean_update = zeros(num_tot,1);
        index_T_update = zeros(length(indx),1);
        
        index_mean_update(id_mean_update,1) = 1;
        index_T_update(id_T_update,1) = 1; 
%         
        
        for j=1:length(id_mean_update)
            mix_Mu{c,1}(id_mean_update(j),1) = init_mu(id_mean_update(j),1);   
            
        end
        %mix_Mu{c,1}(id_mean_update,1) = init_mu;
        
        %if move == 2
            
            
        %need to be modified    
        for j=1:length(id_mean_update_latent)
            
            if id_mean_update_latent(j) == 1
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(1)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(1)))=init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(1)),1); 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(1)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(1))) = log(init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(1)),1));
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(2)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(2)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(2)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(2))) = 0;
         
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(3)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(3)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(3)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(3))) = 0;
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(4)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(4)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(4)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(4))) = 0;
                              
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(5)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(5)))=init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(5)),1); 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(5)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(5))) = log(init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(5)),1));
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(6)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(6)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(6)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(6))) = 0;
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(7)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(7)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(7)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(7))) = 0; 
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(8)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(8)))=init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(8)),1); 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(8)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(8))) = log(init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(8)),1));
 
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(9)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(9)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(9)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(9))) = 0; 
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(10)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(10)))=init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(10)),1); 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(10)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(10))) = log(init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(10)),1));
 
         
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(1)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(1)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(2)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(2)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(3)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(3)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(4)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(4)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(5)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(5)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(6)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(6)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(7)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(7)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(8)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(8)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(9)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(9)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(10)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(10)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(11)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(11)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(12)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(12)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(13)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(13)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(14)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(14)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(15)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(15)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(16)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(16)))=0;
               
               
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(1)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(1)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(2)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(2)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(3)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(3)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(4)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(4)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(5)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(5)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(6)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(6)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(7)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(7)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(8)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(8)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(9)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(9)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(10)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(10)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(11)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(11)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(12)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(12)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(13)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(13)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(14)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(14)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(15)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(15)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(16)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(16)))=0;
 


               
            end
            
            if id_mean_update_latent(j)>1 & id_mean_update_latent(j)<num_states
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(1)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(1)))=init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(1)),1); 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(1)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(1))) = log(init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(1)),1));
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(2)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(2)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(2)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(2))) = 0;
         
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(3)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(3)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(3)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(3))) = 0;
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(4)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(4)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(4)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(4))) = 0;
                              
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(5)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(5)))=init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(5)),1); 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(5)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(5))) = log(init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(5)),1));
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(6)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(6)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(6)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(6))) = 0;
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(7)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(7)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(7)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(7))) = 0; 
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(8)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(8)))=init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(8)),1); 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(8)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(8))) = log(init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(8)),1));
 
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(9)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(9)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(9)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(9))) = 0; 
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(10)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(10)))=init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(10)),1); 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(10)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(10))) = log(init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(10)),1));
                
               if id_mean_update_latent(j)<num_states-1
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(1)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(1)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(2)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(2)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(3)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(3)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(4)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(4)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(5)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(5)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(6)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(6)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(7)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(7)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(8)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(8)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(9)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(9)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(10)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(10)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(11)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(11)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(12)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(12)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(13)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(13)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(14)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(14)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(15)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(15)))=0;
                  mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(16)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(16)))=0;
               
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(1)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(1)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(2)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(2)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(3)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(3)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(4)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(4)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(5)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(5)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(6)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(6)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(7)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(7)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(8)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(8)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(9)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(9)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(10)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(10)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(11)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(11)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(12)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(12)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(13)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(13)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(14)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(14)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(15)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(15)))=0;
                  mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(16)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)+1,1}(16)))=0;
                             
               end 
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(1)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(1)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(2)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(2)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(3)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(3)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(4)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(4)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(5)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(5)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(6)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(6)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(7)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(7)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(8)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(8)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(9)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(9)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(10)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(10)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(11)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(11)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(12)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(12)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(13)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(13)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(14)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(14)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(15)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(15)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(16)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(16)))=0;
               
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(1)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(1)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(2)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(2)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(3)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(3)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(4)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(4)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(5)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(5)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(6)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(6)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(7)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(7)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(8)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(8)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(9)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(9)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(10)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(10)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(11)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(11)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(12)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(12)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(13)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(13)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(14)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(14)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(15)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(15)))=0;
                mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j),1}(16)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j),1}(16)))=0;
               
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(1)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(1)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(2)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(2)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(3)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(3)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(4)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(4)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(5)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(5)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(6)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(6)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(7)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(7)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(8)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(8)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(9)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(9)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(10)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(10)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(11)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(11)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(12)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(12)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(13)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(13)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(14)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(14)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(15)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(15)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(16)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(16)))=0;
               
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(1)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(1)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(2)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(2)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(3)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(3)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(4)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(4)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(5)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(5)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(6)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(6)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(7)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(7)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(8)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(8)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(9)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(9)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(10)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(10)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(11)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(11)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(12)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(12)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(13)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(13)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(14)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(14)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(15)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(15)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(16)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(16)))=0;
               
            end
            
            if id_mean_update_latent(j) == num_states
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(1)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(1)))=init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(1)),1); 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(1)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(1))) = log(init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(1)),1));
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(2)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(2)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(2)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(2))) = 0;
         
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(3)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(3)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(3)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(3))) = 0;
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(4)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(4)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(4)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(4))) = 0;
                              
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(5)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(5)))=init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(5)),1); 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(5)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(5))) = log(init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(5)),1));
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(6)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(6)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(6)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(6))) = 0;
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(7)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(7)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(7)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(7))) = 0; 
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(8)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(8)))=init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(8)),1); 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(8)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(8))) = log(init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(8)),1));
 
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(9)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(9)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(9)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(9))) = 0; 
               
               mix_T{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(10)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(10)))=init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(10)),1); 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(10)),indx_col.J(indx_col.par.Lii{id_mean_update_latent(j),1}(10))) = log(init_T(indx_col.I(indx_col.par.Lii{id_mean_update_latent(j),1}(10)),1));

               
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(1)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(1)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(2)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(2)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(3)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(3)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(4)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(4)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(5)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(5)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(6)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(6)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(7)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(7)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(8)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(8)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(9)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(9)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(10)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(10)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(11)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(11)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(12)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(12)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(13)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(13)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(14)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(14)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(15)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(15)))=0;
               mix_T{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(16)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(16)))=0;
               
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(1)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(1)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(2)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(2)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(3)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(3)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(4)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(4)))=0; 
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(5)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(5)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(6)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(6)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(7)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(7)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(8)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(8)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(9)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(9)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(10)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(10)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(11)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(11)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(12)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(12)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(13)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(13)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(14)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(14)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(15)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(15)))=0;
               mix_T_das{c,1}(indx_col.I(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(16)),indx_col.J(indx_col.par.Ltilde{id_mean_update_latent(j)-1,1}(16)))=0;
               
            end
            
        end
        
        
        
        mix_T{c,1} = sparse(mix_T{c,1});
        mix_T_das{c,1} = sparse(mix_T_das{c,1});
        
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
        
        
%         id_thetaSampled1 = thetaSampled1>limit_cor_up | thetaSampled1<limit_cor_low;
%         %i=1;
%         while sum(sum(id_thetaSampled1))>0
%         %    i=i+1
%             [thetaSampled1,logSampDensPerComp1]=SampleFromMixture_Weight(log_mix_Weights,mix_Mu(1:c),mix_T(1:c),num_samples);
%             id_thetaSampled1 = thetaSampled1>limit_cor_up | thetaSampled1<limit_cor_low;
%         end
        
        [thetaSampled1,logSampDensPerComp1]=SampleFromMixture_Weight(log_mix_Weights,mix_Mu(1:c),mix_T(1:c),num_samples);
        [logRBindicator1,logTotalSampDens1]=CombineMixtureComponents(log_mix_Weights,logSampDensPerComp1); 
        sa = zeros(num_tot,1);
        log_sxxWeights = log_mix_Weights';
        parfor s=1:num_samples
            [lpDens1(s,1),grad1(:,s)] = obtain_grad_param_statespace_sv_Z_higherdim(thetaSampled1(dim_states*num_states+1:dim_states*num_states+num_param,s),...
                thetaSampled1(1:dim_states*num_states,s),y,num_param,num_states,prior,num_some,dim_states,[x1,x2,x3]);
           

            [log_q(s,1),grad_log_q(:,s)] = obtain_grad_q(thetaSampled1(:,s),num_tot,mix_Weights,mix_Mu,mix_T);   
            lwt = logRBindicator1(c,s) - log_sxxWeights(c);
            sa = sa + weightProd(lwt,grad1(:,s)-grad_log_q(:,s));
        end
        grad_mean_mu_temp = sa./num_samples;
        scalar_temp=lpDens1-logTotalSampDens1';
        %grad_mean_mu=((mix_T{c}*mix_T{c}')\grad_mean_mu_temp);
        temp_temp_temp_mu = ((mix_T{c,1})\(grad_mean_mu_temp));
        grad_mean_mu = ((mix_T{c,1}')\temp_temp_temp_mu);
        
        grad_mean_mu_bar = grad_mean_mu;
        
        parfor s=1:num_samples
            %s
            epsilon = randn(num_tot,1);
            theta_param = mix_Mu{c,1} + ((mix_T{c,1}')\epsilon);
%             id_thetaparam = theta_param<=limit_cor_low | theta_param>=limit_cor_up;
%             while sum(sum(id_thetaparam))>0
%                   epsilon = randn(num_tot,1);
%                   theta_param = mix_Mu{c,1} + ((mix_T{c,1}')\epsilon);  
%                   id_thetaparam = theta_param<=limit_cor_low | theta_param>=limit_cor_up;  
%             end
            [log_posterior,grad_log_posterior] = obtain_grad_param_statespace_sv_Z_higherdim(theta_param(dim_states*num_states+1:dim_states*num_states+num_param,1),theta_param(1:dim_states*num_states,1),y,num_param,...
                num_states,prior,num_some,dim_states,[x1,x2,x3]);
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

        
        %while ~stop
        for iter=1:max_iter
            %iter
            %mix_T{c,1}(id_mean_update(1),id_mean_update(1))
            %tlam_mixw
            %phi_weight
            %mix_Mu{2,1}
            max_tlam_mixw=max(max(tlam_mixw),0);
            norm_log_mixw=log1p(sum(exp(tlam_mixw - max_tlam_mixw))-1+exp(-max_tlam_mixw))+max_tlam_mixw;
            log_mix_Weights(1,c-1)=(tlam_mixw-norm_log_mixw)+log(mix_Weights_old(1,c-1)); 
            log_mix_Weights(1,c)=(-norm_log_mixw)+log(mix_Weights_old(1,c-1));
            mix_Weights=exp(log_mix_Weights);
            %tic
            [thetaSampled1,logSampDensPerComp1]=SampleFromMixture_Weight(log_mix_Weights,mix_Mu(1:c),mix_T(1:c),num_samples);
%             id_thetaSampled1 = thetaSampled1>limit_cor_up | thetaSampled1<limit_cor_low;
%         
%             while sum(sum(id_thetaSampled1))>0
%         
%                 [thetaSampled1,logSampDensPerComp1]=SampleFromMixture_Weight(log_mix_Weights,mix_Mu(1:c),mix_T(1:c),num_samples);
%                 id_thetaSampled1 = thetaSampled1>limit_cor_up | thetaSampled1<limit_cor_low;
%             end
            [logRBindicator1,logTotalSampDens1]=CombineMixtureComponents(log_mix_Weights,logSampDensPerComp1); 
            sa = zeros(num_tot,1);
            log_sxxWeights = log_mix_Weights';
            parfor s=1:num_samples
                [lpDens1(s,1),grad1(:,s)] = obtain_grad_param_statespace_sv_Z_higherdim(thetaSampled1(dim_states*num_states+1:dim_states*num_states+num_param,s),thetaSampled1(1:dim_states*num_states,s),y,num_param,...
                    num_states,prior,num_some,dim_states,[x1,x2,x3]);
                [log_q(s,1),grad_log_q(:,s)] = obtain_grad_q(thetaSampled1(:,s),num_tot,mix_Weights,mix_Mu,mix_T);
                lwt = logRBindicator1(c,s) - log_sxxWeights(c);
                sa = sa + weightProd(lwt,grad1(:,s)-grad_log_q(:,s));
            end
            %toc
            grad_mean_mu_temp = sa./num_samples;
            %grad_mean_mu=((mix_T{c}*mix_T{c}')\grad_mean_mu_temp);
            temp_temp_temp_mu = ((mix_T{c,1})\(grad_mean_mu_temp));
            grad_mean_mu = ((mix_T{c,1}')\temp_temp_temp_mu);
            
            temp_temp=(lpDens1-log_q);
            LB_est(iter,1) = mean(temp_temp);
            %LB_est(iter,1)

            if sum(temp_temp>0)==num_samples | sum(temp_temp<0)==num_samples
               [log_sxyWeights,sig]= logMatrixProd(logRBindicator1,(lpDens1-logTotalSampDens1')/num_samples);
               g_m_w2 = exp(log_sxyWeights  - log_sxxWeights) .* sig;
               g_m_w2(sig==0) = 0;
               g_m_w2=g_m_w2';
            else
               g_m_w2=mean(exp(logRBindicator1-log_sxxWeights)'.*(lpDens1-logTotalSampDens1'));
            end
            
            scalar_temp = (lpDens1-log_q);
            %tic
%             [gra_log_q_lambda_T,g_lambda_T_for_c,g_lambda_T]=obtain_grad_with_c_OneAtTimeGeneral_Weight(thetaSampled1,...
%                 mix_Mu,mix_T,num_tot,logRBindicator1,scalar_temp,num_samples,c,c_lambda_T,indx,indx_diag);
% 
%             cov_lambda_T=(1/(num_samples-1))*sum((g_lambda_T_for_c-mean(g_lambda_T_for_c)).*(gra_log_q_lambda_T-mean(gra_log_q_lambda_T)));
%             var_lambda_T=(1/(num_samples-1))*sum((gra_log_q_lambda_T-mean(gra_log_q_lambda_T)).^2);
%             c_lambda_T=cov_lambda_T./var_lambda_T;
            
            
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
%                 id_thetaparam = theta_param<=limit_cor_low | theta_param>=limit_cor_up;
%                 while sum(sum(id_thetaparam))>0
%                       epsilon = randn(num_tot,1);
%                       theta_param = mix_Mu{c,1} + ((mix_T{c,1}')\epsilon);  
%                       id_thetaparam = theta_param<=limit_cor_low | theta_param>=limit_cor_up;  
%                 end
                [log_posterior,grad_log_posterior] = obtain_grad_param_statespace_sv_Z_higherdim(theta_param(dim_states*num_states+1:dim_states*num_states+num_param,1),theta_param(1:dim_states*num_states,1),y,num_param,...
                        num_states,prior,num_some,dim_states,[x1,x2,x3]);
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
 
            delta_T_temp = adapt_alpha_T2*(mt_hat_T{c,1}./(sqrt(vt_hat_T{c,1})+adapt_epsilon)).*index_T_update;
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
            
            %toc
            
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

            %iter=iter+1;

                   
            
        end
        
        
        max_tlam_mixw=max(max(tlam_mixw),0);
        norm_log_mixw=log1p(sum(exp(tlam_mixw - max_tlam_mixw))-1+exp(-max_tlam_mixw))+max_tlam_mixw;
        log_mix_Weights(1,c-1)=(tlam_mixw-norm_log_mixw)+log(mix_Weights_old(1,c-1)); 
        log_mix_Weights(1,c)=(-norm_log_mixw)+log(mix_Weights_old(1,c-1));
        mix_Weights=exp(log_mix_Weights);
        
    end
    
    [measure_latent] = calculating_RKL_higherdim(num_states,num_param,mix_Mu,mix_T,mix_Weights,prior,y,num_some,dim_states,[x1,x2,x3],indx_col);

        
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

       %if rand<prob
       
           id_mean_update_G = [dim_states*num_states+1:dim_states*num_states+num_param]';
           %[init_mu_G] = initialisation_global_parameter(num_states,num_param,mix_Mu,mix_T,mix_Weights,indx,indx_diag,num_chosen,c,prior,y,num_some);
           init_mu_G = mix_Mu{c,1}(dim_states*num_states+1:dim_states*num_states+num_param,1);
           %move = 1;
           
       %else
           %tic
           [id_mean_update_latent,init_mu_latent,init_T]=initialisation_local_update_used(num_states,num_param,mix_Mu,mix_T,mix_Weights,indx,indx_diag,num_chosen,c,prior,y,num_some,dim_states,[x1,x2,x3],indx_col);
           %move=2;
           %toc
           
       %end
       %id_mean_update = [id_mean_update_latent;id_mean_update_latent+num_states;id_mean_update_G];
       id_mean_update = [dim_states*id_mean_update_latent;dim_states*id_mean_update_latent-1;dim_states*id_mean_update_latent-2;dim_states*id_mean_update_latent-3];
       init_mu = [init_mu_latent;init_mu_G];
       
    end
     mix_Weights_old=mix_Weights;
     mod_name=['mixCop_Local',num2str(c),'_',num2str(num_some),'_statespace_sv_Z_regression.mat'];
    save(mod_name,'mix_Weights','mix_Mu','mix_T','mix_T_das','LB_est','mix_Weights_old','id_mean_update_store','measure_latent','LB_smooth','prior');

end

        
        %updating the weights
%         if c>=2
%         m_t_weight_weight = 0;
%         v_t_weight_weight = 0;
% 
%         
%         
%         max_tlam_mixw=max(max(tlam_mixw),0);
%         norm_log_mixw=log1p(sum(exp(tlam_mixw - max_tlam_mixw))-1+exp(-max_tlam_mixw))+max_tlam_mixw;
%         log_mix_Weights(1,1:end-1)=tlam_mixw-norm_log_mixw; 
%         log_mix_Weights(1,end)=-norm_log_mixw;
%         mix_Weights=exp(log_mix_Weights);
%         mix_Weights_actual=[mix_Weights(1,1)*mix_Weights_old,mix_Weights(1,2)];
%         log_mix_Weights_actual=log(mix_Weights_actual); 
        
%         for iter=1:max_iter
%             tlam_mixw_actual = log(mix_Weights_actual(1,1:c-1)) - log( mix_Weights_actual(1,c));
%             [thetaSampledWeight,logSampDensPerCompWeight_q]=SampleFromMixture_Weight(log_mix_Weights_actual,mix_Mu(1:c),mix_T(1:c),num_samples);
%             [logRBindicatorWeight_q,logTotalSampDensWeight_q]=CombineMixtureComponents(log_mix_Weights_actual,logSampDensPerCompWeight_q);
%             
%             for s=1:num_samples
%                 [log_posterior_Weight(s,1),~] = obtain_grad_param_mixNormal(thetaSampledWeight(:,s),num_param,weight_mix_true,mu_true,precs_true);
%                 g_m_w2_weight=(exp(logRBindicatorWeight_q(:,s)-log_mix_Weights_actual')');
%                 grad_weight_ind_weight(s,1:c-1) = -(g_m_w2_weight(1,1:c-1) - g_m_w2_weight(1,c));
%                 
%             end
%             log_weight = log_posterior_Weight - logTotalSampDensWeight_q';
%             max_log_weight = max(log_weight);
%             weight = exp(log_weight - max_log_weight);
%             weight = weight./sum(weight);
%             
%             
%             grad_weight_sum_weight = sum(weight.*grad_weight_ind_weight);
%             m_t_weight_weight=adapt_tau_1*m_t_weight_weight+(1-adapt_tau_1)*grad_weight_sum_weight';
%             v_t_weight_weight=adapt_tau_2*v_t_weight_weight+(1-adapt_tau_2)*grad_weight_sum_weight'.^2;
%             mt_hat_weight_weight=m_t_weight_weight./(1-(adapt_tau_1.^iter));
%             vt_hat_weight_weight=v_t_weight_weight./(1-(adapt_tau_2.^iter));
%             temp_weight_weight=tlam_mixw_actual'-adapt_alpha_weight*(mt_hat_weight_weight./(sqrt(vt_hat_weight_weight)+adapt_epsilon));
%             
%             if sum(isnan(temp_weight_weight))>0
%                %theta_phi_weight = theta_phi_weight;
%                 tlam_mixw_actual = tlam_mixw_actual;
%             else
%                 tlam_mixw_actual = temp_weight_weight';
%                %theta_phi_weight = temp_weight;    
%             end
% 
%             max_tlam_mixw_actual=max(max(tlam_mixw_actual),0);
%             norm_log_mixw_actual=log1p(sum(exp(tlam_mixw_actual - max_tlam_mixw_actual))-1+exp(-max_tlam_mixw_actual))+max_tlam_mixw_actual;
%             log_mix_Weights_actual(1,1:end-1)=tlam_mixw_actual-norm_log_mixw_actual; 
%             log_mix_Weights_actual(1,end)=-norm_log_mixw_actual;
%             mix_Weights_actual = exp(log_mix_Weights_actual);
% 
%         end
% 
%         
%        end






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
