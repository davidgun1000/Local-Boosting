%this code reproduce figures for random effect logistic regression model. 
% To run this need to add functions folder to the path.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure 2 in the main paper
figure
for i=1:20
    load(['mixCop_Global',num2str(i),'_500_reparam_MixT_Bimodal.mat']);
    mean_collect(i,1) = mean(measure_latent);
end
subplot(1,3,1);plot(mean_collect,'LineWidth',3);xlabel({'number of components'},'FontSize',20);ylabel('$\widetilde{s}[K]$', 'Interpreter', 'latex', 'FontSize', 20);set(gca,'FontSize',20);title('mixT(i)','FontSize',20);
hold on
for i=1:20
    load(['mixCop_Global',num2str(i),'_500_reparam_mixT_VS.mat']);
    mean_collect(i,1) = mean(measure_latent);
end
subplot(1,3,2);plot(mean_collect,'LineWidth',3);xlabel({'number of components'},'FontSize',20);ylabel('$\widetilde{s}[K]$', 'Interpreter', 'latex', 'FontSize', 20);title('mixT(ii)','FontSize',20);
set(gca,'FontSize',20)
hold on
for i=1:20
    load(['mixCop_Global',num2str(i),'_500_reparam_Z.mat']);
    mean_collect(i,1) = mean(measure_latent);
end
subplot(1,3,3);plot(mean_collect,'LineWidth',3);xlabel({'number of components'},'FontSize',20);ylabel('$\widetilde{s}[K]$', 'Interpreter', 'latex', 'FontSize', 20);title('Z','FontSize',20);
set(gca,'FontSize',20)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure 3 of main paper
nrSamples=100000;
num_points_ks=1000;
load('HMC_MixT_full1_Bimodal.mat');
i=2;
load(['mixCop_Global',num2str(i),'_500_reparam_mixT_Bimodal.mat']);
for k = 1:length(mix_Mu)
    mix_Precs{k,1} = mix_T{k,1}*mix_T{k,1}';
end
theta_cop1_MixT_Bimodal = SampleFromMixture_plot(log(mix_Weights),mix_Mu,mix_Precs,nrSamples);

load('wd_polypharm__tcop_Bimodal.mat');
[theta_tcop_Bimodal]= VArand_tcop(nrSamples,eta,mu,tkappa,l,log_nu,'');

load('wd_polypharm_YJ_tcop_Bimodal.mat');
[theta_tcop_YJ_Bimodal]= VArand_tcop(nrSamples,eta,mu,tkappa,l,log_nu,'YJ');

load('wd_polypharm_YJdouble_tcop_Bimodal.mat');
[theta_tcop_doubleYJ_Bimodal] = VArand_tcop(nrSamples,eta,mu,tkappa,l,log_nu,'YJdouble');

marginal = 10;
subplot(2,3,1);
[f,xi]=ksdensity(Post1.theta(:,marginal),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
hold on
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_YJ_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_doubleYJ_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
title(['b_{10}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-2.5 2.5 0 Inf])

marginal = 20;
subplot(2,3,2);
[f,xi]=ksdensity(Post1.theta(:,marginal),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
hold on
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_YJ_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_doubleYJ_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
title(['b_{20}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-2.5 2.5 0 Inf])

marginal = 30;
subplot(2,3,3);
[f,xi]=ksdensity(Post1.theta(:,marginal),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
hold on
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_YJ_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_doubleYJ_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
title(['b_{30}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-2.5 2.5 0 Inf])

marginal = 10;
subplot(2,3,4);
[f,xi]=ksdensity(Post1.theta(:,marginal),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
hold on
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_YJ_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_doubleYJ_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
title(['b_{10}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-2.5 2.5 -10 Inf])

marginal = 20;
subplot(2,3,5);
[f,xi]=ksdensity(Post1.theta(:,marginal),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
hold on
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_YJ_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_doubleYJ_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
title(['b_{20}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-2.5 2.5 -10 Inf])

marginal = 30;
subplot(2,3,6);
[f,xi]=ksdensity(Post1.theta(:,marginal),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
hold on
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_YJ_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
[f,xi]=ksdensity(theta_tcop_doubleYJ_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
title(['b_{30}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-2.5 2.5 -10 Inf])
legend({'HMC','MixGau','tcop','YJ-tcop','D-YJ-tcop'},'FontSize',20)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure S2 of the online supplement
nrSamples=100000;
num_points_ks=1000;
load('HMC_MixT_full1_Bimodal.mat');
i=2;
load(['mixCop_Global',num2str(i),'_500_reparam_mixT_Bimodal.mat']);
for k = 1:length(mix_Mu)
    mix_Precs{k,1} = mix_T{k,1}*mix_T{k,1}';
end
theta_cop1_MixT_Bimodal = SampleFromMixture_plot(log(mix_Weights),mix_Mu,mix_Precs,nrSamples);

load('wd_polypharm__tcop_Bimodal.mat');
[theta_tcop_Bimodal]= VArand_tcop(nrSamples,eta,mu,tkappa,l,log_nu,'');

load('wd_polypharm_YJ_tcop_Bimodal.mat');
[theta_tcop_YJ_Bimodal]= VArand_tcop(nrSamples,eta,mu,tkappa,l,log_nu,'YJ');

load('wd_polypharm_YJdouble_tcop_Bimodal.mat');
[theta_tcop_doubleYJ_Bimodal] = VArand_tcop(nrSamples,eta,mu,tkappa,l,log_nu,'YJdouble');

figure
set(gcf,'Position',[100 100 1400 650]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Common axis limits: change these to zoom in/out
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xLim = [-2.3 2.3];
yLim = [-2.3 2.3];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number of grid points and contour levels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nGrid = 200;
nContour = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel 1: MixGau
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,1);

x = theta_cop1_MixT_Bimodal(10,:)';
y = theta_cop1_MixT_Bimodal(20,:)';

x1 = linspace(xLim(1), xLim(2), nGrid);
x2 = linspace(yLim(1), yLim(2), nGrid);
[X1, X2] = meshgrid(x1, x2);

f = ksdensity([x y], [X1(:) X2(:)]);
f = reshape(f, size(X1));

contour(X1, X2, f, nContour, 'Color', 'r', 'LineWidth', 1);
xlim(xLim);
ylim(yLim);
xlabel('b_{10}','FontSize',20);
ylabel('b_{20}','FontSize',20);
title('MixGau');
set(gca,'FontSize',20);
box on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel 2: HMC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,2);

x = Post1.theta(:,10);
y = Post1.theta(:,20);

x1 = linspace(xLim(1), xLim(2), nGrid);
x2 = linspace(yLim(1), yLim(2), nGrid);
[X1, X2] = meshgrid(x1, x2);

f = ksdensity([x y], [X1(:) X2(:)]);
f = reshape(f, size(X1));

contour(X1, X2, f, nContour, 'Color', 'r', 'LineWidth', 1);
xlim(xLim);
ylim(yLim);
xlabel('b_{10}','FontSize',20);
ylabel('b_{20}','FontSize',20);
title('HMC');
set(gca,'FontSize',20);
box on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel 3: YJ-tcop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,3);

x = theta_tcop_YJ_Bimodal(10,:)';
y = theta_tcop_YJ_Bimodal(20,:)';

x1 = linspace(xLim(1), xLim(2), nGrid);
x2 = linspace(yLim(1), yLim(2), nGrid);
[X1, X2] = meshgrid(x1, x2);

f = ksdensity([x y], [X1(:) X2(:)]);
f = reshape(f, size(X1));

contour(X1, X2, f, nContour, 'Color', 'r', 'LineWidth', 1);
xlim(xLim);
ylim(yLim);
xlabel('b_{10}','FontSize',20);
ylabel('b_{20}','FontSize',20);
title('YJ-tcop');
set(gca,'FontSize',20);
box on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Panel 4: D-YJ-tcop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,4);

x = theta_tcop_doubleYJ_Bimodal(10,:)';
y = theta_tcop_doubleYJ_Bimodal(20,:)';

x1 = linspace(xLim(1), xLim(2), nGrid);
x2 = linspace(yLim(1), yLim(2), nGrid);
[X1, X2] = meshgrid(x1, x2);

f = ksdensity([x y], [X1(:) X2(:)]);
f = reshape(f, size(X1));

contour(X1, X2, f, nContour, 'Color', 'r', 'LineWidth', 1);
xlim(xLim);
ylim(yLim);
xlabel('b_{10}','FontSize',20);
ylabel('b_{20}','FontSize',20);
title('D-YJ-tcop');
set(gca,'FontSize',20);
box on;


% figure
% subplot(2,2,1);
% x = theta_cop1_MixT_Bimodal(10,:)';
% y = theta_cop1_MixT_Bimodal(20,:)';
% 
% % % Define grid
% x1 = linspace(min(x), max(x), 100);
% x2 = linspace(min(y), max(y), 100);
% [X1, X2] = meshgrid(x1, x2);
% % 
% % % Evaluate kernel density
% f = ksdensity([x y], [X1(:) X2(:)]);
% f = reshape(f, size(X1));
% % % Contour plot
% contour(X1, X2, f, 'levels',1,'color','r');
% axis([-3 3 -3 3 0 Inf ])
% xlabel('b_{10}','FontSize',20); ylabel('b_{20}','FontSize',20);
% title('MixGau');
% 
% subplot(2,2,2);
% x = Post1.theta(:,10);
% y = Post1.theta(:,20);
% 
% % % Define grid
% x1 = linspace(min(x), max(x), 100);
% x2 = linspace(min(y), max(y), 100);
% [X1, X2] = meshgrid(x1, x2);
% % 
% % % Evaluate kernel density
% f = ksdensity([x y], [X1(:) X2(:)]);
% f = reshape(f, size(X1));
% % % Contour plot
% contour(X1, X2, f, 'levels',1,'color','r');
% axis([-3 3 -3 3 0 Inf ])
% xlabel('b_{10}','FontSize',20); ylabel('b_{20}','FontSize',20);
% title('HMC');
% 
% subplot(2,2,3);
% x = theta_tcop_YJ_Bimodal(10,:)';
% y = theta_tcop_YJ_Bimodal(20,:)';
% 
% % % Define grid
% x1 = linspace(min(x), max(x), 100);
% x2 = linspace(min(y), max(y), 100);
% [X1, X2] = meshgrid(x1, x2);
% % 
% % % Evaluate kernel density
% f = ksdensity([x y], [X1(:) X2(:)]);
% f = reshape(f, size(X1));
% % % Contour plot
% contour(X1, X2, f, 'levels',1,'color','r');
% axis([-3 3 -3 3 0 Inf ])
% xlabel('b_{10}','FontSize',20); ylabel('b_{20}','FontSize',20);
% title('YJ-tcop');
% 
% subplot(2,2,4);
% x = theta_tcop_doubleYJ_Bimodal(10,:)';
% y = theta_tcop_doubleYJ_Bimodal(20,:)';
% 
% % % Define grid
% x1 = linspace(min(x), max(x), 100);
% x2 = linspace(min(y), max(y), 100);
% [X1, X2] = meshgrid(x1, x2);
% % 
% % % Evaluate kernel density
% f = ksdensity([x y], [X1(:) X2(:)]);
% f = reshape(f, size(X1));
% % % Contour plot
% contour(X1, X2, f, 'levels',1,'color','r');
% axis([-3 3 -3 3 0 Inf ])
% xlabel('b_{10}','FontSize',20); ylabel('b_{20}','FontSize',20);
% title('D-YJ-tcop');


% figure
% subplot(1,2,1);
% scatter(theta_tcop_Bimodal(10,:)',theta_tcop_Bimodal(20,:)')
% hold on
% scatter(Post1.theta(:,10),Post1.theta(:,20))
% scatter(theta_cop1_MixT_Bimodal(10,:)',theta_cop1_MixT_Bimodal(20,:)');
% scatter(theta_tcop_YJ_Bimodal(10,:)',theta_tcop_YJ_Bimodal(20,:)')
% scatter(theta_tcop_doubleYJ_Bimodal(10,:)',theta_tcop_doubleYJ_Bimodal(20,:)')
% 
% subplot(1,2,2);
% scatter(theta_tcop_Bimodal(20,:)',theta_tcop_Bimodal(30,:)')
% hold on
% scatter(Post1.theta(:,20),Post1.theta(:,30))
% scatter(theta_cop1_MixT_Bimodal(20,:)',theta_cop1_MixT_Bimodal(30,:)');
% scatter(theta_tcop_YJ_Bimodal(20,:)',theta_tcop_YJ_Bimodal(30,:)')
% scatter(theta_tcop_doubleYJ_Bimodal(20,:)',theta_tcop_doubleYJ_Bimodal(30,:)')
% legend({'tcop','HMC','MixGau','YJ-tcop','D-YJ-tcop'},'FontSize',20)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure S1 of the online supplement
nrSamples=100000;
num_points_ks=1000;
load('HMC_MixT_full1_VS.mat');
i=5;
load(['mixCop_Global',num2str(i),'_500_reparam_mixT_VS.mat']);
for k = 1:length(mix_Mu)
    mix_Precs{k,1} = mix_T{k,1}*mix_T{k,1}';
end
theta_cop1_MixT_VS = SampleFromMixture_plot(log(mix_Weights),mix_Mu,mix_Precs,nrSamples);

load('wd_polypharm__tcop_VS.mat');
[theta_tcop_VS]= VArand_tcop(nrSamples,eta,mu,tkappa,l,log_nu,'');

load('wd_polypharm_YJ_tcop_VS.mat');
[theta_tcop_YJ_VS]= VArand_tcop(nrSamples,eta,mu,tkappa,l,log_nu,'YJ');

load('wd_polypharm_YJdouble_tcop_VS.mat');
[theta_tcop_doubleYJ_VS] = VArand_tcop(nrSamples,eta,mu,tkappa,l,log_nu,'YJdouble');

figure
set(gcf,'Position',[100 100 1600 800]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_points_ks = 300;

marginals = [10 20 30];

method_names = {'HMC','MixGau','tcop','YJ-tcop','D-YJ-tcop'};

% Line styles and colours
line_styles = {'-','--',':','-.','-'};
line_width_top = 2.5;
line_width_bottom = 2.0;

colors = [
    0.0000 0.0000 0.0000;   % HMC: black
    0.0000 0.4470 0.7410;   % MixGau: blue
    0.8500 0.3250 0.0980;   % tcop: orange/red
    0.4660 0.6740 0.1880;   % YJ-tcop: green
    0.4940 0.1840 0.5560    % D-YJ-tcop: purple
];

font_size = 16;

% Top panels: use a tighter x-axis range
% Increase these percentiles to zoom in more.
top_prct = [5 95];

% Bottom log-density panels: use a wider range, but not too wide.
% This shows tail behaviour without making the figure too busy.
bottom_prct = [2.5 97.5];

% Log-density floor for bottom panels
% Larger values, e.g. -7, make the bottom panels less busy.
% Smaller values, e.g. -12, show more tail detail.
log_floor = -12;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you want manual x-limits, set this to true
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

use_manual_xlim = false;

manual_xlim_top = [
    -3 3;    % b10
    -3 3;    % b20
    -3 3     % b30
];

manual_xlim_bottom = [
    -6 6;    % b10
    -6 6;    % b20
    -6 6     % b30
];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Layout
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

legend_handles = gobjects(5,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main loop over marginals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for jj = 1:length(marginals)

    marginal = marginals(jj);

    % Extract draws
    draws_HMC      = Post1_MixT_VS.theta(:,marginal);
    draws_MixGau   = theta_cop1_MixT_VS(marginal,:)';
    draws_tcop     = theta_tcop_VS(marginal,:)';
    draws_YJ_tcop  = theta_tcop_YJ_VS(marginal,:)';
    draws_DYJ_tcop = theta_tcop_doubleYJ_VS(marginal,:)';

    draws_all_methods = {
        draws_HMC;
        draws_MixGau;
        draws_tcop;
        draws_YJ_tcop;
        draws_DYJ_tcop
    };

    % Remove non-finite values
    for k = 1:5
        temp = draws_all_methods{k};
        draws_all_methods{k} = temp(isfinite(temp));
    end

    all_draws = vertcat(draws_all_methods{:});

    % Automatic x-limits
    if use_manual_xlim
        xlim_top = manual_xlim_top(jj,:);
        xlim_bottom = manual_xlim_bottom(jj,:);
    else
        xlim_top = prctile(all_draws, top_prct);
        pad_top = 0.10 * range(xlim_top);
        xlim_top = [xlim_top(1)-pad_top, xlim_top(2)+pad_top];

        xlim_bottom = prctile(all_draws, bottom_prct);
        pad_bottom = 0.10 * range(xlim_bottom);
        xlim_bottom = [xlim_bottom(1)-pad_bottom, xlim_bottom(2)+pad_bottom];
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Top row: ordinary density
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nexttile(jj);

    xi_top = linspace(xlim_top(1), xlim_top(2), num_points_ks);

    hold on

    for k = 1:5
        f = ksdensity(draws_all_methods{k}, xi_top);

        h = plot(xi_top, f, ...
            'LineWidth', line_width_top, ...
            'LineStyle', line_styles{k}, ...
            'Color', colors(k,:));

        if jj == 1
            legend_handles(k) = h;
        end
    end

    hold off

    title(['$b_{' num2str(marginal) '}$: density'], ...
        'Interpreter','latex','FontSize',font_size);

    xlabel(['$b_{' num2str(marginal) '}$'], ...
        'Interpreter','latex','FontSize',font_size);

    if jj == 1
        ylabel('Density','FontSize',font_size);
    end

    xlim(xlim_top);
    ylim([0 Inf]);

    set(gca,'FontSize',font_size);
    box on
    grid on

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Bottom row: log-density
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nexttile(jj+3);

    xi_bottom = linspace(xlim_bottom(1), xlim_bottom(2), num_points_ks);

    hold on

    for k = 1:5
        f = ksdensity(draws_all_methods{k}, xi_bottom);

        % Avoid log(0) and suppress noisy extreme tails
        log_f = log(max(f, exp(log_floor)));

        plot(xi_bottom, log_f, ...
            'LineWidth', line_width_bottom, ...
            'LineStyle', line_styles{k}, ...
            'Color', colors(k,:));
    end

    hold off

    title(['$b_{' num2str(marginal) '}$: log-density'], ...
        'Interpreter','latex','FontSize',font_size);

    xlabel(['$b_{' num2str(marginal) '}$'], ...
        'Interpreter','latex','FontSize',font_size);

    if jj == 1
        ylabel('Log-density','FontSize',font_size);
    end

    xlim(xlim_bottom);
    ylim([log_floor Inf]);

    set(gca,'FontSize',font_size);
    box on
    grid on

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Single legend outside the panels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lgd = legend(legend_handles, method_names, ...
    'FontSize',14, ...
    'Location','eastoutside');

lgd.Layout.Tile = 'east';


% marginal = 10;
% subplot(2,3,1);
% [f,xi]=ksdensity(Post1_MixT_VS.theta(:,marginal),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% hold on
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_YJ_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_doubleYJ_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% title(['b_{10}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-10 10 0 Inf])
% 
% marginal = 20;
% subplot(2,3,2);
% [f,xi]=ksdensity(Post1_MixT_VS.theta(:,marginal),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% hold on
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_YJ_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_doubleYJ_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% title(['b_{20}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-10 10 0 Inf])
% 
% marginal = 30;
% subplot(2,3,3);
% [f,xi]=ksdensity(Post1_MixT_VS.theta(:,marginal),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% hold on
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_YJ_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_doubleYJ_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% title(['b_{30}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-10 10 0 Inf])
% 
% marginal = 10;
% subplot(2,3,4);
% [f,xi]=ksdensity(Post1_MixT_VS.theta(:,marginal),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% hold on
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_YJ_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_doubleYJ_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% title(['b_{10}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-20 20 -10 Inf])
% 
% marginal = 20;
% subplot(2,3,5);
% [f,xi]=ksdensity(Post1_MixT_VS.theta(:,marginal),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% hold on
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_YJ_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_doubleYJ_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% title(['b_{20}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-20 20 -10 Inf])
% 
% marginal = 30;
% subplot(2,3,6);
% [f,xi]=ksdensity(Post1_MixT_VS.theta(:,marginal),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% hold on
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_YJ_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% [f,xi]=ksdensity(theta_tcop_doubleYJ_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% title(['b_{30}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-20 20 -10 Inf])
% legend({'HMC','MixGau','tcop','YJ-tcop','D-YJ-tcop'},'FontSize',20)


%% Figure 4 of the main paper.

nrSamples=100000;
num_points_ks=1000;
load('HMC_Z_full.mat');
i=7;
%i=14;
load(['mixCop_Global',num2str(i),'_500_reparam_Z.mat']);
for k = 1:length(mix_Mu)
    mix_Precs{k,1} = mix_T{k,1}*mix_T{k,1}';
end
theta_cop1_Z = SampleFromMixture_plot(log(mix_Weights),mix_Mu,mix_Precs,nrSamples);

load('wd_polypharm__tcop_Z.mat');
[theta_tcop_Z]= VArand_tcop(nrSamples,eta,mu,tkappa,l,log_nu,'');

load('wd_polypharm_YJ_tcop_Z.mat');
[theta_tcop_YJ_Z]= VArand_tcop(nrSamples,eta,mu,tkappa,l,log_nu,'YJ');

load('wd_polypharm_YJdouble_tcop_Z.mat');
[theta_tcop_doubleYJ_Z] = VArand_tcop(nrSamples,eta,mu,tkappa,l,log_nu,'YJdouble');

marginal = 10;
subplot(2,3,1);
[f,xi]=ksdensity(Post_Z.theta(:,marginal),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
hold on
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_YJ_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_doubleYJ_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
title(['b_{10}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-10 10 0 Inf])

marginal = 20;
subplot(2,3,2);
[f,xi]=ksdensity(Post_Z.theta(:,marginal),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
hold on
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_YJ_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_doubleYJ_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
title(['b_{20}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-10 10 0 Inf])

marginal = 30;
subplot(2,3,3);
[f,xi]=ksdensity(Post_Z.theta(:,marginal),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
hold on
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_YJ_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_doubleYJ_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
title(['b_{30}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-10 10 0 Inf])

marginal = 10;
subplot(2,3,4);
[f,xi]=ksdensity(Post_Z.theta(:,marginal),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
hold on
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_YJ_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_doubleYJ_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
title(['b_{10}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-20 20 -10 Inf])

marginal = 20;
subplot(2,3,5);
[f,xi]=ksdensity(Post_Z.theta(:,marginal),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
hold on
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_YJ_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_doubleYJ_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
title(['b_{20}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-20 20 -10 Inf])

marginal = 30;
subplot(2,3,6);
[f,xi]=ksdensity(Post_Z.theta(:,marginal),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
hold on
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_YJ_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
[f,xi]=ksdensity(theta_tcop_doubleYJ_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
title(['b_{30}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-20 20 -10 Inf])
legend({'HMC','MixGau','tcop','YJ-tcop','D-YJ-tcop'},'FontSize',20)

%% Figure 5 of the main paper
i=2;
load(['mixCop_Global',num2str(i),'_50_reparam_MixT_Bimodal.mat']);
LB_Global_Bimodal = LB_est;

i=2;
load(['mixCop_Local',num2str(i),'_some_50_reparam_MixT_Bimodal.mat']);
LB_Local_Bimodal = LB_est;

i=2;
load(['mixCop_Global',num2str(i),'_50_reparam_MixT_VS.mat']);
LB_Global_VS = LB_est;

i=2;
load(['mixCop_Local',num2str(i),'_some_50_reparam_MixT_VS.mat']);
LB_Local_VS = LB_est;

i=2;
load(['mixCop_Global',num2str(i),'_50_reparam_Z.mat']);
LB_Global_Z = LB_est;

i=2;
load(['mixCop_Local',num2str(i),'_some_50_reparam_Z.mat']);
LB_Local_Z = LB_est;

figure
subplot(3,1,1);
plot(LB_Global_Bimodal,'LineWidth',2.5);
hold on
plot(LB_Local_Bimodal,'LineWidth',2.5);

subplot(3,1,2);
plot(LB_Global_VS,'LineWidth',2.5)
hold on
plot(LB_Local_VS,'LineWidth',2.5);

subplot(3,1,3);
plot(LB_Global_Z,'LineWidth',2.5)
hold on
plot(LB_Local_Z,'LineWidth',2.5);
legend({'Global','Local'},'FontSize',20)

%% Figure 6 of the main paper
figure
load('mixCop_Local20_some_50_reparam_MixT_Bimodal.mat');
for i=2:20
    id_mean_update_store{i,1}(:,2) = i;
end

for i=2:20
    subplot(3,2,1);scatter(id_mean_update_store{i,1}(:,2),id_mean_update_store{i,1}(:,1),'o');xlabel('boosting iterations','FontSize',20);ylabel('selected index','FontSize',20);set(gca,'FontSize',20);
    hold on
end

for i=1:20
    load(['mixCop_Local',num2str(i),'_some_50_reparam_MixT_Bimodal.mat']);
    mean_collect(i,1) = mean(measure_latent);
end
subplot(3,2,2);plot(mean_collect,'LineWidth',3);xlabel({'number of components'},'FontSize',20);ylabel('$\widetilde{s}[K]$', 'Interpreter', 'latex', 'FontSize', 20);set(gca,'FontSize',20);title('mixT(i)','FontSize',20);

load('mixCop_Local20_some_50_reparam_MixT_VS.mat');
for i=2:20
    id_mean_update_store{i,1}(:,2) = i;
end
for i=2:20
    subplot(3,2,3);scatter(id_mean_update_store{i,1}(:,2),id_mean_update_store{i,1}(:,1),'o');xlabel('boosting iterations','FontSize',20);ylabel('selected index','FontSize',20);set(gca,'FontSize',20);
    hold on
end

for i=1:20
    load(['mixCop_Local',num2str(i),'_some_50_reparam_MixT_VS.mat']);
    mean_collect(i,1) = mean(measure_latent);
end
subplot(3,2,4);plot(mean_collect,'LineWidth',3);xlabel({'number of components'},'FontSize',20);ylabel('$\widetilde{s}[K]$', 'Interpreter', 'latex', 'FontSize', 20);set(gca,'FontSize',20);title('mixT(ii)','FontSize',20);

load('mixCop_Local20_some_50_reparam_Z.mat');
for i=2:20
    id_mean_update_store{i,1}(:,2) = i;
end
for i=2:20
    subplot(3,2,5);scatter(id_mean_update_store{i,1}(:,2),id_mean_update_store{i,1}(:,1),'o');xlabel('boosting iterations','FontSize',20);ylabel('selected index','FontSize',20);set(gca,'FontSize',20);
    hold on
end

for i=1:20
    load(['mixCop_Local',num2str(i),'_some_50_reparam_Z.mat']);
    mean_collect(i,1) = mean(measure_latent);
end
subplot(3,2,6);plot(mean_collect,'LineWidth',3);xlabel({'number of components'},'FontSize',20);ylabel('$\widetilde{s}[K]$', 'Interpreter', 'latex', 'FontSize', 20);set(gca,'FontSize',20);title('Z','FontSize',20);

%%

% figure
% for i=1:20
%     load(['mixCop_Local',num2str(i),'_some_500_reparam_MixT_Bimodal.mat']);
%     mean_collect(i,1) = mean(measure_latent);
% end
% subplot(1,3,1);plot(mean_collect,'LineWidth',3);xlabel({'number of components'},'FontSize',20);ylabel({'{s}[K]'},'FontSize',20);set(gca,'FontSize',20);title('mixT(i)','FontSize',20);
% 
% for i=1:50
%     load(['mixCop_Local',num2str(i),'_some_500_reparam_MixT_VS.mat']);
%     mean_collect(i,1) = mean(measure_latent);
% end
% subplot(1,3,2);plot(mean_collect,'LineWidth',3);xlabel({'number of components'},'FontSize',20);ylabel({'{s}[K]'},'FontSize',20);set(gca,'FontSize',20);title('mixT(ii)','FontSize',20);
% 
% for i=1:50
%     load(['mixCop_Local',num2str(i),'_some_500_reparam_Z.mat']);
%     mean_collect(i,1) = mean(measure_latent);
% end
% subplot(1,3,3);plot(mean_collect,'LineWidth',3);xlabel({'number of components'},'FontSize',20);ylabel({'{s}[K]'},'FontSize',20);set(gca,'FontSize',20);title('Z','FontSize',20);








%% Figure S3 of the online supplement

nrSamples=100000;
num_points_ks=1000;
i=2;
load(['mixCop_Local',num2str(i),'_some_50_reparam_MixT_Bimodal.mat']);
for k = 1:length(mix_Mu)
    mix_Precs{k,1} = mix_T{k,1}*mix_T{k,1}';
end
theta_cop1_MixT_Bimodal = SampleFromMixture_plot(log(mix_Weights),mix_Mu,mix_Precs,nrSamples);

marginal = 10;
subplot(2,4,1);
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
title(['b_{10}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-2.5 2.5 0 Inf])

marginal = 20;
subplot(2,4,2);
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
title(['b_{20}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-2.5 2.5 0 Inf])

marginal = 30;
subplot(2,4,3);
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
title(['b_{30}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-2.5 2.5 0 Inf])

marginal = 60;
subplot(2,4,4);
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
title(['b_{60}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-4 4 0 Inf])

marginal = 10;
subplot(2,4,5);
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
title(['b_{10}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-2.5 2.5 -10 Inf])

marginal = 20;
subplot(2,4,6);
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
title(['b_{20}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-2.5 2.5 -10 Inf])

marginal = 30;
subplot(2,4,7);
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2);
title(['b_{30}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-2.5 2.5 -10 Inf])

marginal = 60;
subplot(2,4,8);
[f,xi]=ksdensity(theta_cop1_MixT_Bimodal(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',2);
title(['b_{60}: MixT(i)'],'FontSize',20);
set(gca,'FontSize',20);
axis([-4 4 0 Inf])

%% Figure S4 of the online supplement

nrSamples=100000;
num_points_ks=1000;
i=4;
load(['mixCop_Local',num2str(i),'_some_50_reparam_MixT_VS.mat']);
for k = 1:length(mix_Mu)
    mix_Precs{k,1} = mix_T{k,1}*mix_T{k,1}';
end
theta_cop1_MixT_VS = SampleFromMixture_plot(log(mix_Weights),mix_Mu,mix_Precs,nrSamples);

figure
set(gcf,'Position',[100 100 1700 750]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

marginals = [10 20 30 60];
num_marginals = length(marginals);

num_points_ks = 400;

font_size = 20;
line_width = 3;

% Top-row density panels:
% Use central part of the distribution only.
% Increase these values to zoom in more, e.g. [5 95].
top_prct = [2.5 97.5];

% Bottom-row log-density panels:
% Use a wider range than the top row, but not too wide.
bottom_prct = [0.5 99.5];

% Minimum log-density shown in the bottom panels.
% Larger value = cleaner plot; smaller value = more tail detail.
log_floor = -8;

% Add padding around automatic x-limits
pad_top = 0.15;
pad_bottom = 0.10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optional manual x-axis limits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

use_manual_xlim = true;

% If use_manual_xlim = true, edit these manually.
manual_xlim_top = [
    -0.5 0.5;     % b10
    -0.5 0.5;     % b20
    -0.5 0.5;     % b30
    -4.0 4.0      % b60
];

manual_xlim_bottom = [
    -2 2;         % b10
    -2 2;         % b20
    -2 2;         % b30
    -6 6          % b60
];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Layout
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tiledlayout(2,4,'TileSpacing','compact','Padding','compact');

for jj = 1:num_marginals

    marginal = marginals(jj);

    % Extract draws
    draws = theta_cop1_MixT_VS(marginal,:)';
    draws = draws(isfinite(draws));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Automatic x-limits
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if use_manual_xlim

        xlim_top = manual_xlim_top(jj,:);
        xlim_bottom = manual_xlim_bottom(jj,:);

    else

        % Top row: tight central region
        xlim_top = prctile(draws, top_prct);
        range_top = xlim_top(2) - xlim_top(1);

        if range_top == 0
            range_top = std(draws);
        end

        if range_top == 0
            range_top = 1;
        end

        xlim_top = [
            xlim_top(1) - pad_top * range_top, ...
            xlim_top(2) + pad_top * range_top
        ];

        % Bottom row: wider region, but still avoids extreme noisy tails
        xlim_bottom = prctile(draws, bottom_prct);
        range_bottom = xlim_bottom(2) - xlim_bottom(1);

        if range_bottom == 0
            range_bottom = std(draws);
        end

        if range_bottom == 0
            range_bottom = 1;
        end

        xlim_bottom = [
            xlim_bottom(1) - pad_bottom * range_bottom, ...
            xlim_bottom(2) + pad_bottom * range_bottom
        ];

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Top row: density
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nexttile(jj);

    xi_top = linspace(xlim_top(1), xlim_top(2), num_points_ks);
    f_top = ksdensity(draws, xi_top);

    plot(xi_top, f_top, 'LineWidth', line_width);

    title(['$b_{' num2str(marginal) '}$: MixT(ii)'], ...
        'Interpreter','latex','FontSize',font_size);

    xlabel(['$b_{' num2str(marginal) '}$'], ...
        'Interpreter','latex','FontSize',font_size);

    if jj == 1
        ylabel('Density','FontSize',font_size);
    end

    xlim(xlim_top);
    ylim([0 Inf]);

    set(gca,'FontSize',font_size);
    box on;
    grid on;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Bottom row: log-density
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nexttile(jj + num_marginals);

    xi_bottom = linspace(xlim_bottom(1), xlim_bottom(2), num_points_ks);
    f_bottom = ksdensity(draws, xi_bottom);

    log_f = log(f_bottom);

    % Remove extremely small estimated densities to reduce noisy tails
    log_f(log_f < log_floor) = NaN;

    plot(xi_bottom, log_f, 'LineWidth', line_width);

    title(['$b_{' num2str(marginal) '}$: log-density'], ...
        'Interpreter','latex','FontSize',font_size);

    xlabel(['$b_{' num2str(marginal) '}$'], ...
        'Interpreter','latex','FontSize',font_size);

    if jj == 1
        ylabel('Log-density','FontSize',font_size);
    end

    xlim(xlim_bottom);
    ylim([log_floor Inf]);

    set(gca,'FontSize',20);
    box on;
    grid on;

end

% marginal = 10;
% subplot(2,4,1);
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% title(['b_{10}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-10 10 0 Inf])
% 
% marginal = 20;
% subplot(2,4,2);
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% title(['b_{20}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-10 10 0 Inf])
% 
% marginal = 30;
% subplot(2,4,3);
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% title(['b_{30}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-10 10 0 Inf])
% 
% marginal = 60;
% subplot(2,4,4);
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,(f),'LineWidth',3);
% title(['b_{60}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-10 10 0 Inf])
% 
% marginal = 10;
% subplot(2,4,5);
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% title(['b_{10}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-20 20 -10 Inf])
% 
% marginal = 20;
% subplot(2,4,6);
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% title(['b_{20}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-20 20 -10 Inf])
% 
% marginal = 30;
% subplot(2,4,7);
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% title(['b_{30}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-20 20 -10 Inf])
% 
% marginal = 60;
% subplot(2,4,8);
% [f,xi]=ksdensity(theta_cop1_MixT_VS(marginal,:),'NumPoints',num_points_ks);
% plot(xi,log(f),'LineWidth',3);
% title(['b_{60}: MixT(ii)'],'FontSize',20);
% set(gca,'FontSize',20);
% axis([-20 20 -10 Inf])

%% Figure S5 of the online supplement

nrSamples=100000;
num_points_ks=1000;
i=7;
load(['mixCop_Local',num2str(i),'_some_50_reparam_Z.mat']);
for k = 1:length(mix_Mu)
    mix_Precs{k,1} = mix_T{k,1}*mix_T{k,1}';
end
theta_cop1_Z = SampleFromMixture_plot(log(mix_Weights),mix_Mu,mix_Precs,nrSamples);

marginal = 10;
subplot(2,4,1);
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
title(['b_{10}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-10 10 0 Inf])

marginal = 20;
subplot(2,4,2);
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
title(['b_{20}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-10 10 0 Inf])

marginal = 30;
subplot(2,4,3);
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
title(['b_{30}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-10 10 0 Inf])

marginal = 60;
subplot(2,4,4);
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,(f),'LineWidth',3);
title(['b_{60}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-10 10 0 Inf])

marginal = 10;
subplot(2,4,5);
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
title(['b_{10}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-20 20 -10 Inf])

marginal = 20;
subplot(2,4,6);
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
title(['b_{20}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-20 20 -10 Inf])

marginal = 30;
subplot(2,4,7);
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
title(['b_{30}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-20 20 -10 Inf])

marginal = 60;
subplot(2,4,8);
[f,xi]=ksdensity(theta_cop1_Z(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',3);
title(['b_{60}: Z'],'FontSize',20);
set(gca,'FontSize',20);
axis([-20 20 -10 Inf])

%% Figure 1 of the main paper
i=2;
load(['mixCop_Global',num2str(i),'_500_reparam_MixT_Bimodal.mat']);
LB_Global_Bimodal = LB_est;

i=2;
load(['mixCop_Global',num2str(i),'_500_CV_MixT_Bimodal.mat']);
LB_CV_Bimodal = LB_est;

i=2;
load(['mixCop_Global',num2str(i),'_500_reparam_MixT_VS.mat']);
LB_Global_VS = LB_est;

i=2;
load(['mixCop_Global',num2str(i),'_500_CV_MixT_VS.mat']);
LB_CV_VS = LB_est;

i=2;
load(['mixCop_Global',num2str(i),'_500_reparam_Z.mat']);
LB_Global_Z = LB_est;

i=2;
load(['mixCop_Global',num2str(i),'_500_CV_Z.mat']);
LB_CV_Z = LB_est;

figure
subplot(3,1,1);
plot(LB_CV_Bimodal,'LineWidth',2.5);
hold on
plot(LB_Global_Bimodal,'LineWidth',2.5);

subplot(3,1,2);
plot(LB_CV_VS,'LineWidth',2.5)
hold on
plot(LB_Global_VS,'LineWidth',2.5);

subplot(3,1,3);
plot(LB_CV_Z,'LineWidth',2.5)
hold on
plot(LB_Global_Z,'LineWidth',2.5);
legend({'CV','Reparam.'},'FontSize',20);

%% Figure S6 of the online supplement

i=20;
load(['mixCop_Global',num2str(i),'_500_reparam_MixT_Bimodal.mat']);
t = (1:1:20)';
subplot(1,3,1);plot(t,cpu_time/60,'LineWidth',2);
set(gca,'FontSize',20);
xlabel('number of components','FontSize',20);
ylabel('Time','FontSize',20);
title({'MixT(i)'},'FontSize',20);
axis([0 20 8 18])

i=20;
load(['mixCop_Global',num2str(i),'_500_reparam_MixT_VS.mat']);
subplot(1,3,2);plot(t,cpu_time/60,'LineWidth',2);
set(gca,'FontSize',20);
xlabel('number of components','FontSize',20);
ylabel('Time','FontSize',20);
title({'MixT(ii)'},'FontSize',20);
axis([0 20 8 18])


i=20;
load(['mixCop_Global',num2str(i),'_500_reparam_Z.mat']);
subplot(1,3,3);plot(t,cpu_time/60,'LineWidth',2);
set(gca,'FontSize',20);
xlabel('number of components','FontSize',20);
ylabel('Time','FontSize',20);
title({'Z'},'FontSize',20);
axis([0 20 8 18])

