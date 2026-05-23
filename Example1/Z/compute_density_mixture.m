function [logTotSampDens] = compute_density_mixture(theta,dim_y,mix_Weights,mix_mu,mix_T)

    nrComponents = length(mix_mu);
    k = length(mix_mu{1,1});
    %cholPrec = cell(nrComponents,1);
    for c=1:nrComponents
        cholPrec{c} = (mix_T{c})';
    end
    %logSampDensPerComp = zeros(nrComponents,1);
    for c=1:nrComponents
        if k>1
           logSampDensPerComp(c,:) = (-(dim_y/2)*log(2*pi)+sum(log(diag(cholPrec{c}))) ...
            -0.5*sum((cholPrec{c}*(theta-mix_mu{c})).^2));
        else
           logSampDensPerComp(c,:) = (-(dim_y/2)*log(2*pi)+sum(log(diag(cholPrec{c}))) ...
            -0.5*((cholPrec{c}*(theta-mix_mu{c})).^2)); 

        end
    end
    [~,logTotSampDens]=CombineMixtureComponents(log(mix_Weights),logSampDensPerComp);



end