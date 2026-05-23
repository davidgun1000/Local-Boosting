function [f, logf, dlogf, df] = zpdf_general(alpha, mu, s, a, b)
% ZPDF_GENERAL  Z (logit–Beta/EGB2) pdf, log-pdf, and derivatives (general a,b).
%   [f, logf, dlogf, df] = ZPDF_GENERAL(alpha, mu, s, a, b)
%
% Inputs:
%   alpha : array of evaluation points
%   mu    : location (scalar or array, broadcastable)
%   s     : scale > 0
%   a,b   : shape parameters > 0
%
% Outputs:
%   f     : pdf at alpha
%   logf  : log-pdf at alpha
%   dlogf : d/d(alpha) log-pdf (score)
%   df    : d/d(alpha) pdf
%
% Identities with u=(alpha-mu)/s:
%   logf = -logB(a,b) - log s - b*u - (a+b)*softplus(-u)
%   dlogf = (1/s) * [ a - (a+b)*sigmoid(u) ]
%   f = exp(logf),  df = f .* dlogf

%     arguments
%         alpha
%         mu     (1,1) double = 0
%         s      (1,1) double {mustBePositive} = 0.01
%         a      (1,1) double {mustBePositive} = 0.5
%         b      (1,1) double {mustBePositive} = 0.5
%     end

    % Broadcast u
    u = (alpha - mu) ./ s;

    % Stable softplus(x) = log(1+exp(x))
    softplus = @(x) max(0,x) + log1p(exp(-abs(x)));

    % log Beta
    logB = gammaln(a) + gammaln(b) - gammaln(a + b);

    % log-pdf (stable)
    logf = -logB - log(s) - b.*u - (a + b).*softplus(-u);

    % pdf
    f = exp(logf);

    % Stable sigmoid via tanh
    sigmoid = 0.5 .* (1 + tanh(u/2));

    % derivative of log-pdf (score)
    dlogf = (1./s) .* (a - (a + b) .* sigmoid);

    % derivative of pdf
    df = f .* dlogf;
end