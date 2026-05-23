function [logTotalSampDens,grad_param]=obtain_grad_q(theta,dim_y,mix_Weights,mix_mu,mix_T)
nrComponents = length(mix_mu);
k = length(mix_mu{1,1});
%cholPrec = cell(nrComponents,1);
for c=1:nrComponents
    cholPrec{c} = (mix_T{c})';
end
logSampDensPerComp = zeros(nrComponents,1);
for c=1:nrComponents
    if k>1
       logSampDensPerComp(c,:) = (-(dim_y/2)*log(2*pi)+sum(log(diag(cholPrec{c}))) ...
        -0.5*sum((cholPrec{c}*(theta-mix_mu{c})).^2));
    else
       logSampDensPerComp(c,:) = (-(dim_y/2)*log(2*pi)+sum(log(diag(cholPrec{c}))) ...
        -0.5*((cholPrec{c}*(theta-mix_mu{c})).^2)); 
    
    end
end
[logRBindicator,logTotalSampDens]=CombineMixtureComponents(log(mix_Weights),logSampDensPerComp);

if nargout>1
gn = zeros(size(theta,1),1);
for j=1:length(mix_mu)
    %gn = \nabla_z log q(z)
    gn = gn + weightProd(logRBindicator(j),(mix_T{j}*mix_T{j}')*(mix_mu{j}-theta));
end
grad_param=gn;
end


end


%     temp1=-((df_t+dim_y)/2);
%     u=inv(1+((theta-mu_t)'*(sigma_t\(theta-mu_t)))/df_t);
%     v=(2/df_t)*(sigma_t\(theta-mu_t));
% 
%     grad_param=temp1.*u.*v;
%     u_das=-((1+((theta-mu_t)'*(sigma_t\(theta-mu_t)))/df_t).^(-2))*((2/df_t)*(sigma_t\(theta-mu_t)));
%     v_das=(2/df_t)*inv(sigma_t);
%     
%     hess_temp=temp1*(u_das*v+u*v_das);
%     hess_param=diag(hess_temp);