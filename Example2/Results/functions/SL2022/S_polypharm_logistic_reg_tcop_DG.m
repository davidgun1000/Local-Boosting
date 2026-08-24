clear
Tranformations = {'' 'YJ' 'YJdouble' 'iGH'};
for i = 1:4
    close all
    clearvars -except Tranformations i
    rng(19900209);
    workingDir = pwd;
    addpath(genpath([workingDir '/Derivatives']));
    %Defining the type of approximation
    p = 5;                                                                     % Number of factors on the covariance matrix of the approximation
    Transf = Tranformations{i};                                                             % Type of transformation. For Yeo-Johnson transformation set to 'YJ'. For GH transformation set to 'GH', for no transformation set to '' or 'None'
    niter = 10000;                                                              % Number of total iterations for the VB algorithm
    %------ Defining log-posteriors and derivatives of the model---------------
    %load('FamaFrenchData.mat')
    load('Zdata.mat');
    num_param=1;
    dim_states=4;
    %y = AgricRF_stan;
    %x1 = MktRF_stan;
    %x2 = SMB_stan;
    %x3 = HML_stan;
    x=[x1,x2,x3];
    length_data = length(y); % length of data
    num_states = length(y); 
    num_some = num_states;
    num_tot = dim_states*num_states+num_param;
    b_n = 9; %first index of the random effect
    prior.mu = 0; %hyperparameters for the priors
    prior.s = 0.1;
    prior.a = 0.5;
    prior.b = 0.5;
    prior.hp_sig2=100;
    prior.sig2=1;
    logpost = @(theta) log_posterior_polypharm(y,x,theta,num_states,prior,dim_states,num_some);
    VBtransfobj = initial_values_polypharm_tcop(Transf,num_tot,p);                   %Setting initial values for variational parameters
    %--------------------------------------------------------------------------
    %--------------------------------------------------------------------------
    [mu,l,tkappa,eta,log_nu,LB,StoreTime] = VBtransf_tcop(logpost,num_tot,p,Transf,niter,100,false,VBtransfobj);
    save(['wd_polypharm_' Transf '_tcop_StateSpace_Z_sim.mat'],'mu','l','tkappa','eta','log_nu','LB','StoreTime')
end