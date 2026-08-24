samplesStruct.HMC   = theta_exact;    % [N x 3]
samplesStruct.MixGau       = theta_rvgaw;    % [N x 3]
samplesStruct.Gau = theta_whittle;  % [N x 3]

paramNames = {'\phi','\sigma_\eta','\sigma_\epsilon'};
trueVals   = [0.9, 0.7, 0.5];  % set [] if unknown

cornerKDEContours(samplesStruct, paramNames, trueVals);