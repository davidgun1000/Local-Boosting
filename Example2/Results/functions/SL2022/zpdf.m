function y = zpdf(x, a, b, mu, s)
%ZPDF  Pdf of the Z (logit–Beta) distribution.
%   y = ZPDF(x, a, b)
%   y = ZPDF(x, a, b, mu, s)

    if nargin < 4 || isempty(mu), mu = 0; end
    if nargin < 5 || isempty(s),  s  = 1; end
    y = exp(zlogpdf(x, a, b, mu, s));   % stable via log-pdf
end