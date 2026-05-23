% sample from the approximate posterior
function [xSampled,logSampDensPerComp] = SampleFromMixture(logMixWeights,mixMeans,mixPrecs,nrSamples,gen_sample,theta_old)
k = length(mixMeans{1});
rawNorm = randn(k,nrSamples);
rawUnif = rand(nrSamples,1);
[xSampled,logSampDensPerComp] = SampleFromMixtureHelper(logMixWeights,mixMeans,mixPrecs,rawNorm,rawUnif,gen_sample,theta_old);
end
