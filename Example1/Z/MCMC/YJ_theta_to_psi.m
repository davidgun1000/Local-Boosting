function [psi]=YJ_theta_to_psi(theta,gam_YJ)

    id_neg=theta<0;
    psi_num(id_neg,1)=-(((-theta(id_neg,1)+1).^(2-gam_YJ(id_neg,1)))-1);
    psi_den(id_neg,1)=2-gam_YJ(id_neg,1);
    psi(id_neg,1)=psi_num(id_neg,1)./psi_den(id_neg,1); 

    id_pos=theta>0;
    psi(id_pos,1)=(((theta(id_pos,1)+1).^gam_YJ(id_pos,1))-1)./gam_YJ(id_pos,1);  
%     if theta<0
%        psi_num=-(((-theta+1).^(2-gam_YJ))-1);
%        psi_den=2-gam_YJ;
%        psi=psi_num./psi_den; 
%     else
%        psi=(((theta+1).^gam_YJ)-1)./gam_YJ;  
%     end




end