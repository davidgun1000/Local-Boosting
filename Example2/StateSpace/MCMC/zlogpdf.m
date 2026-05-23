function y = zlogpdf(x, a, b, mu, s)
%ZLOGPDF  Log-pdf of the Z (logit–Beta) distribution.
%   y = ZLOGPDF(x, a, b)             % standard Z (mu=0, s=1)
%   y = ZLOGPDF(x, a, b, mu, s)      % location-scale: Y = mu + s*Z
%
%   a>0, b>0, s>0. Vectorized over x.

    if nargin < 4 || isempty(mu), mu = 0; end
    if nargin < 5 || isempty(s),  s  = 1; end
    if any(a <= 0, 'all') || any(b <= 0, 'all') || any(s <= 0, 'all')
        error('Parameters must satisfy a>0, b>0, s>0.');
    end

    z = (x - mu) ./ s;

    % log(sigmoid(z)) = -softplus(-z),  log(1 - sigmoid(z)) = -softplus(z)
    log_sig   = -softplus(-z);
    log_1msig = -softplus( z);

    y = a .* log_sig + b .* log_1msig - betaln(a,b) - log(s);
end

% ---- helpers ----
function y = softplus(u)
% numerically stable log(1+exp(u))
    y = zeros(size(u));
    idx = u > 0;
    y(idx)  = u(idx) + log1p(exp(-u(idx)));
    y(~idx) = log1p(exp(u(~idx)));
end