% sample from the approximate posterior
function [xSampled,logSampDensPerComp] = SampleFromMixture_plot(logMixWeights,mixMeans,mixPrecs,nrSamples)
k = length(mixMeans{1});
rawNorm = randn(k,nrSamples);
rawUnif = rand(nrSamples,1);
[xSampled,logSampDensPerComp] = SampleFromMixtureHelper_plot(logMixWeights,mixMeans,mixPrecs,rawNorm,rawUnif);
end
