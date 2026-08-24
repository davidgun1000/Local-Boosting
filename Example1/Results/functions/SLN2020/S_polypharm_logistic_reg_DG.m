clear
Tranformations = {'' 'YJ' 'YJdouble' 'iGH'};
for i = 1:4
    close all
    clearvars -except Tranformations i

    rng(19900209);
    workingDir = pwd;
    addpath(genpath([workingDir '/Derivatives']));
    %Defining the type of approximation
    p = 5;                                                                  % Number of factors on the covariance matrix of the approximation
    Transf = Tranformations{i};                                             % Type of transformation. Yeo-Johnson =  'YJ'. iGH transformation =  'iGH'. No transformation =  ''. Double Yeo-Johnson = 'YJdouble'
    niter = 10000;  
    % Number of total iterations for the VB algorithm
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
    prior.sig2_alpha = [0.0001,0.0001];
    prior.mu = [-2,2];
    prior.w_a = [0.5,0.5];
    prior.df = 3;
    num_randeffect = length(unique(data(:,2))); % number of random effects
    num_period = length(data(:,1))/num_randeffect;
    num_some= num_randeffect;
    logpost = @(theta) log_posterior_polypharm(y,x,theta,prior,num_period,group,num_some);
    %--------------------------------------------------------------------------
    %figure
    %--------------------------------------------------------------------------
    [mu,d,B,eta,LB,StoreTime] = VBtransf(logpost,num_tot,p,Transf,niter,100,false,[]);
    save(['wd_polypharm_' Transf '_Bimodal.mat'],'mu','d','B','eta','LB','StoreTime','Transf','prior');
end

