function [theta]=YJ_psi_to_theta(psi,gam_YJ)

    id_neg=psi<0;
    theta(id_neg,1)=1-((1-psi(id_neg,1).*(2-gam_YJ(id_neg,1))).^(1./(2-gam_YJ(id_neg,1)))); 

    id_pos=psi>0;
    theta(id_pos,1)=((1+psi(id_pos,1).*gam_YJ(id_pos,1)).^(1./gam_YJ(id_pos,1)))-1;

    %if psi<0
    %   theta=1-((1-psi.*(2-gam_YJ)).^(1./(2-gam_YJ))); 
    %else
    %    theta=((1+psi.*gam_YJ).^(1./gam_YJ))-1;
    %end


end