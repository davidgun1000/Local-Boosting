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
    load('Data/polypharm.mat','Xtr','Ytr')
    q = size(Xtr,2) + 1;
    b_n = 9; %first index of the random effect
    VBtransfobj = initial_values_polypharm_tcop(Transf,q,p);                   %Setting initial values for variational parameters
    logpost = @(theta) log_posterior_polypharm(theta,Ytr,Xtr,q,b_n);
    %--------------------------------------------------------------------------
    figure
    %--------------------------------------------------------------------------
    [mu,l,tkappa,eta,log_nu,LB,StoreTime] = VBtransf_tcop(logpost,q,p,Transf,niter,100,false,VBtransfobj);
    save(['wd_polypharm_' Transf '_tcop.mat'],'mu','l','tkappa','eta','log_nu','LB','StoreTime')
end