ksdensity(theta_cop1_MixMF(2,:))
hold on
ksdensity(theta_cop1_MixN(2,:))
ksdensity(Post.theta(:,2))
legend({'MF','Precs','HMC'},'FontSize',20)


[f,xi]=ksdensity(theta_cop1_MixMF(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2.5)
hold on;
[f,xi]=ksdensity(theta_cop1_MixN(marginal,:),'NumPoints',num_points_ks);
plot(xi,log(f),'LineWidth',2.5)
legend({'MF','Precs'},'FontSize',20)
axis([-10 10 -20 Inf])