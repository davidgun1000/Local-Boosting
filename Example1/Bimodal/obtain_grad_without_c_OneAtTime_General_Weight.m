function [gra_log_q_lambda_T,g_lambda_T]=obtain_grad_without_c_OneAtTime_General_Weight(thetaSampled,mix_Mu,mix_T,num_param,logRBindicator,scalar_temp,nrSamples2,c,indx,indx_diag)
    
        parfor j=1:nrSamples2
            grad_T = sparse(diag(1./diag(mix_T{c,1}))) - (thetaSampled(:,j)-mix_Mu{c,1})*((thetaSampled(:,j)-mix_Mu{c,1})'*mix_T{c,1});
            temp_vec=grad_T(sub2ind(size(grad_T),[indx(:,1)'],[indx(:,2)']));
            temp_vec_diag=grad_T(sub2ind(size(grad_T),[indx_diag(:,1)'],[indx_diag(:,2)'])).*mix_T{c,1}(sub2ind(size(mix_T{c,1}),[indx_diag(:,1)'],[indx_diag(:,2)']));
            temp_mat=zeros(num_param);
            temp_mat(sub2ind(size(temp_mat),indx(:,1),indx(:,2)))=temp_vec';
            temp_mat(sub2ind(size(temp_mat),indx_diag(:,1),indx_diag(:,2)))=temp_vec_diag';
            grad_T = temp_mat(sub2ind(size(temp_mat),indx(:,1),indx(:,2)));
            gra_log_q_lambda_T(j,:)= exp(logRBindicator(c,j)).*grad_T;
            g_lambda_T(j,:) = gra_log_q_lambda_T(j,:).*scalar_temp(j,1);
            
        end
        
%        ( diag(1./diag(mix_T{c,1})) - (thetaSampled(:,s)-mix_Mu{c,1})*(thetaSampled(:,s)-mix_Mu{c,1})'*mix_T{c,1}');
        
end


% function [gra_log_q_lambda_B,g_lambda_B,gra_log_q_lambda_d,g_lambda_d]=obtain_grad_without_c_OneAtTime_General(psiSampled,mix_Mu,mix_B,mix_d,num_param,num_factor_VB,logRBindicator,scalar_temp,nrSamples2,c)
%     sum_B = abs(sum(mix_B{2,1}));
%     if sum_B>0
%         for j=1:nrSamples2
%             Dm2=sparse(1:num_param,1:num_param,1./(mix_d{c,1}.^2),num_param,num_param);
%             Sigmainv=Dm2-Dm2*mix_B{c,1}/(sparse(1:num_factor_VB,1:num_factor_VB,1,num_factor_VB,num_factor_VB)+mix_B{c,1}'*Dm2*mix_B{c,1})*mix_B{c,1}'*Dm2;
%             SigmainvB=Sigmainv*mix_B{c,1};
%             SigmainvD=Sigmainv*diag([mix_d{c,1}]);
%             gra_log_q_lambda_B(j,:)=exp(logRBindicator(2,j)).*(B2vechB(-SigmainvB+Sigmainv*(psiSampled(:,j)-mix_Mu{c,1})*(psiSampled(:,j)-mix_Mu{c,1})'*SigmainvB,num_factor_VB));
%             g_lambda_B(j,:)=gra_log_q_lambda_B(j,:).*scalar_temp(j,1);
%             gra_log_q_lambda_d(j,:)=exp(logRBindicator(2,j)).*(diag(-SigmainvD+Sigmainv*(psiSampled(:,j)-mix_Mu{c,1})*(psiSampled(:,j)-mix_Mu{c,1})'*SigmainvD));
%             g_lambda_d(j,:)=gra_log_q_lambda_d(j,:).*scalar_temp(j,1);    
%         end
%     else
%         gra_log_q_lambda_B=[];
%         g_lambda_B=[];
%         for j=1:nrSamples2
%             Dm2=sparse(1:num_param,1:num_param,1./(mix_d{c,1}.^2),num_param,num_param);
%             Sigmainv=Dm2;
%             SigmainvD=Sigmainv*diag([mix_d{c,1}]);
%             gra_log_q_lambda_d(j,:)=exp(logRBindicator(2,j)).*(diag(-SigmainvD+Sigmainv*(psiSampled(:,j)-mix_Mu{c,1})*(psiSampled(:,j)-mix_Mu{c,1})'*SigmainvD));
%             g_lambda_d(j,:)=gra_log_q_lambda_d(j,:).*scalar_temp(j,1);
%         end
%         
%         
%         
%     end
%     
% end