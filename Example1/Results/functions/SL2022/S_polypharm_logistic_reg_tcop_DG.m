clear
Tranformations = {'' 'YJ' 'YJdouble' 'iGH'};
for i = 1:4
    close all
    clearvars -except Tranformations i
    %rng(19900209);
    workingDir = pwd;
    addpath(genpath([workingDir '/Derivatives']));
    %Defining the type of approximation
    p = 5;                                                                     % Number of factors on the covariance matrix of the approximation
    Transf = Tranformations{i};                                                             % Type of transformation. For Yeo-Johnson transformation set to 'YJ'. For GH transformation set to 'GH', for no transformation set to '' or 'None'
    niter = 10000;                                                              % Number of total iterations for the VB algorithm
    %------ Defining log-posteriors and derivatives of the model---------------
    load('polypharm_data.mat')
    n = length(data(:,1));
    
    q = size(data(:,3:end),2)+1;
    num_randeffect = length(unique(data(:,2))); % number of random effects
    num_tot = num_randeffect+q;
    b_n = 9; %first index of the random effect
    x = [ones(n,1),data(:,3:end)]; %matrix of predictors
    y = data(:,1); %response variable
    group = data(:,2); 
    prior.sig2_beta = 1;
    prior.sig2_alpha = [0.01,0.01];
    prior.mu = [-2,2];
    prior.w_a = [0.5,0.5];
    prior.df = 1000;
    num_randeffect = length(unique(data(:,2))); % number of random effects
    num_period = length(data(:,1))/num_randeffect;
    num_some= num_randeffect;
    logpost = @(theta) log_posterior_polypharm(y,x,theta,prior,num_period,group,num_some);
    VBtransfobj = initial_values_polypharm_tcop(Transf,num_tot,p);                   %Setting initial values for variational parameters
    %--------------------------------------------------------------------------
    %--------------------------------------------------------------------------
    [mu,l,tkappa,eta,log_nu,LB,StoreTime] = VBtransf_tcop(logpost,num_tot,p,Transf,niter,100,false,VBtransfobj);
    save(['wd_polypharm_' Transf '_tcop_Bimodal.mat'],'mu','l','tkappa','eta','log_nu','LB','StoreTime','prior')
end