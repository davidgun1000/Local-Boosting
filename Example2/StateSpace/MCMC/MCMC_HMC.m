%HMC method for logistic regression with random effects example, where the
%priors of random effects follow Z distribution.

load('FamaFrenchData.mat');
y = FoodRF_stan;
x1 = MktRF_stan;
x2 = SMB_stan;
x3 = HML_stan;
num_param=1;
dim_states = 4; %dimension of state variables
length_data = length(y); % length of data
num_states = length(y); 
num_tot = dim_states*num_states+num_param;
prior.mu = 0; %hyperparameters for the priors
prior.s = 0.1;
prior.a = 0.5;
prior.b = 0.5;
prior.hp_sig2=100;
prior.sig2=1;
num_some= num_states;
burn=5000;
nit=100000;
s=burn+nit;
theta = 0.1*ones(num_tot,1); 
[log_posterior,grad_log_posterior] = obtain_grad_param_statespace_sv_Z_higherdim(theta(dim_states*num_states+1:dim_states*num_states+num_param,1),theta(1:dim_states*num_states,1),y,num_param,...
                num_states,prior,num_some,dim_states,[x1,x2,x3]); %computing log h and grad of h
%post = Mix_Normal_PDF(theta,dim_y,mix_Weights_true,mix_mu_true,mix_sigma_true,mix_precs_true);

D1=num_tot;
V1=0.001*eye(D1);
scale=0.1;
target_accept=0.20;
accept=0;

pstar=0.44;
alpha=-norminv(pstar/2);
n0=round(5/(pstar*(1-pstar)));
dim_sigma=1;
sigma_1=1;
sigma2_1=sigma_1^2;
sigma_vec_1=sigma_1;

f=@(theta) obtain_grad_param_statespace_sv_Z_higherdim(theta(dim_states*num_states+1:dim_states*num_states+num_param,1),theta(1:dim_states*num_states,1),y,num_param,...
                num_states,prior,num_some,dim_states,[x1,x2,x3]);
epsilon=0.005;

gamma=0.05;
t0=10;
kappa=0.75;
mu=log(10*epsilon);
epsilonbar=1;
nwarmup=1000;
epsilon_seq=zeros(nwarmup,1);
epsilonbar_seq=zeros(nwarmup,1);
epsilon_seq(1,1)=epsilon;
Hbar=0;
delta=0.8;




k=1;

for i=1:100
    i
    tic
    %accept
    %theta(1,1)
    A1 = rand();    
    [logp0,grad_beta0]=f(theta);
    [theta_star,ave_alpha,~,~,~,depth]=NUTS_mixnom_logistic(f,epsilon,theta,logp0,grad_beta0,100,eye(num_tot),eye(num_tot));
    %depth
    theta=theta_star; 
    if i<=burn
        etax=1/(i+t0);
        Hbar=(1-etax)*Hbar+etax*(delta-ave_alpha);
        epsilon_adapt=exp(mu-sqrt(i)/gamma*Hbar);
        epsilon_seq(i)=epsilon_adapt;
        etan=i^(-kappa);
        epsilonbar=exp((1-etan)*log(epsilonbar)+etan*log(epsilon_adapt));
        epsilonbar_seq(i,1)=epsilonbar;
        epsilon=epsilonbar;
    else
        epsilon=epsilonbar;
    end
    
    
    
    Post.theta(i,:) = theta;
    if mod(i,1000)==0
       save('/scratch/hc87/dg2271/HMC_Z_StateSpace_Food.mat','Post','prior');
 
        
    end
    
%     if i>burn
%        Post.theta(k,:)=theta;
%        k=k+1;
%     end
        
   
     cpu_time(i,1) = toc;
end
save('/scratch/hc87/dg2271/HMC_Z_StateSpace_Food.mat','Post','prior');
% theta_col1 = Post.theta(1:100000,:);
% theta_col2 = Post.theta(100001:200000,:);
% theta_col3 = Post.theta(200001:300000,:);
% theta_col4 = Post.theta(300001:400000,:);
% theta_col5 = Post.theta(400001:500000,:);
% theta_col6 = Post.theta(500001:600000,:);
% theta_col7 = Post.theta(600001:700000,:);
% theta_col8 = Post.theta(700001:800000,:);
% theta_col9 = Post.theta(800001:900000,:);
% theta_col10 = Post.theta(900001:end,:);
% 
% save('MCMC_MixNom_HMC1.mat','theta_col1');
% save('MCMC_MixNom_HMC2.mat','theta_col2');
% save('MCMC_MixNom_HMC3.mat','theta_col3');
% save('MCMC_MixNom_HMC4.mat','theta_col4');
% save('MCMC_MixNom_HMC5.mat','theta_col5');
% save('MCMC_MixNom_HMC6.mat','theta_col6');
% save('MCMC_MixNom_HMC7.mat','theta_col7');
% save('MCMC_MixNom_HMC8.mat','theta_col8');
% save('MCMC_MixNom_HMC9.mat','theta_col9');
% save('MCMC_MixNom_HMC10.mat','theta_col10');
% 
