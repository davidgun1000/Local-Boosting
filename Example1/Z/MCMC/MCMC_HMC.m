%HMC method for logistic regression with random effects example, where the
%priors of random effects follow Z distribution.

load("polypharm_data.mat")
num_param=size(data(:,3:end),2)+1;
num_randeffect = length(unique(data(:,2)));
num_tot = num_randeffect+num_param;
num_period = length(data(:,1))/num_randeffect;
prior.sig2_beta = 1; %hyperparameters for the priors
prior.mean = 0;
prior.s = 0.1;
prior.a = 0.5;
prior.b = 0.5;
y = data(:,1);
group = data(:,2);
n = length(data(:,1));
x = [ones(n,1),data(:,3:end)];
num_some= num_randeffect;

burn=5000;
nit=100000;
s=burn+nit;
theta = 0.1*ones(num_tot,1); 
[log_posterior,grad_log_posterior] = obtain_grad_logposterior_logit_some(y,x,theta(num_randeffect+1:num_randeffect+num_param,1),theta(1:num_randeffect,1),...
                prior,num_period,group,num_some);
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

f=@(theta) obtain_grad_logposterior_logit_some(y,x,theta(num_randeffect+1:num_randeffect+num_param,1),theta(1:num_randeffect,1),...
                prior,num_period,group,num_some);
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
    depth
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
       save('HMC_Z_full.mat','Post');
 
        
    end
    
%     if i>burn
%        Post.theta(k,:)=theta;
%        k=k+1;
%     end
        
   
     cpu_time(i,1)=toc;
end
save('HMC_Z_full.mat','Post');
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
