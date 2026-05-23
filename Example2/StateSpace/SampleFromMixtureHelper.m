% sample from GMM
function [xSampled,logSampDensPerComp] = SampleFromMixtureHelper(logMixWeights,mixMeans,mixPrecs,rawNorm,rawUnif,gen_sample,theta_old)

nrSamples = size(rawUnif,1);

% dimensions
k = length(mixMeans{1});
nrComponents = length(mixMeans);
%nrComponentsActual = length(mixMeans);
% get cholesky's of the precision matrices of all Gaussian components
% cholPrec = cell(nrComponents,1);
% for c=1:nrComponents
%     cholPrec{c} = (mixPrecs{c}');
% end

cholPrec = cell(nrComponents,1);
for c=1:nrComponents
    cholPrec{c} = (mixPrecs{c})';
end


%mix_Weights=exp(logMixWeights);
%mix_Weights_actual=[mix_Weights(1,1)*mix_Weights_old,mix_Weights(1,2)];
%logMixWeightsactual=log(mix_Weights_actual);
%log_mix_Weights_old=log(mix_Weights_old);

% sample mixture component indicators
comps = SampleMixture_comps(logMixWeights, nrSamples, rawUnif);
assert( all(size(rawNorm) == [k, nrSamples]) );

% sample from the Gaussian mixture components

if gen_sample ==1
z = rawNorm;

for j=1:nrSamples
    xSampled(:,j) = mixMeans{comps(j)} + cholPrec{comps(j)}\z(:,j);
end
else
    xSampled = theta_old;
    
end

% get densities at the sampled points
%logSampDensPerCompActual = zeros(nrComponentsActual,nrSamples);
logSampDensPerComp = zeros(nrComponents,nrSamples);

for c=1:nrComponents
    if k>1
    logSampDensPerComp(c,:) = (-(k/2)*log(2*pi)+sum(log(diag(cholPrec{c}))) ...
        -0.5*sum((cholPrec{c}*(xSampled-repmat(mixMeans{c},1,nrSamples))).^2));
    else
    logSampDensPerComp(c,:) = (-(k/2)*log(2*pi)+sum(log(diag(cholPrec{c}))) ...
        -0.5*((cholPrec{c}*(xSampled-repmat(mixMeans{c},1,nrSamples))).^2));
    end
end

% temp=logSampDensPerCompActual(1:nrComponentsActual-1,:) + repmat(log_mix_Weights_old',1,nrSamples);
% max_log=max(temp,[],1);
% total_temp=log1p(sum(exp(temp - max_log),1)-1) + max_log;
% 
% logSampDensPerComp(1,:)=total_temp;
% logSampDensPerComp(2,:)=logSampDensPerCompActual(nrComponentsActual,:);

end

function comps = SampleMixture_comps(llp, nrSamples, rawUnif)
%generate the mixture index according to the log probability weigths (llp)

n=length(llp);
assert(size(llp,1) == 1);
assert(size(llp,2) == n);

if n==1
    comps=ones(nrSamples,1);
else
    assert( all(size(rawUnif) == [nrSamples,1] ) );

    [nllp,ind]=sort(llp,'descend');
    cnllp = zeros(n,1);
    cnllp(1) = nllp(1);
    logsum=nllp(1);
    for i=2:n
        cnllp(i)=log1p(exp(nllp(i)-logsum))+logsum;
        logsum=cnllp(i);
    end
    cpb = (cnllp-logsum)';
    ss=1+sum(repmat(log(rawUnif),1,n)>repmat(cpb,nrSamples,1),2);
    comps = arrayfun(@(x) ind(x), ss);
end
end


