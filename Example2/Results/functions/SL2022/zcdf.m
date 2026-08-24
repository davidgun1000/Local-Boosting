function F = zcdf(x, a, b, mu, s)
%ZCDF  Cdf of the Z (logit–Beta) distribution (regularized incomplete Beta).
%   F = ZCDF(x, a, b)
%   F = ZCDF(x, a, b, mu, s)

    if nargin < 4 || isempty(mu), mu = 0; end
    if nargin < 5 || isempty(s),  s  = 1; end
    if any(a <= 0, 'all') || any(b <= 0, 'all') || any(s <= 0, 'all')
        error('Parameters must satisfy a>0, b>0, s>0.');
    end

    z = (x - mu) ./ s;
    p = sigmoid_stable(z);        % p = 1./(1+exp(-z)), computed stably
    F = betainc(p, a, b);         % regularized incomplete Beta = CDF
end

% ---- helper ----
function p = sigmoid_stable(u)
% numerically stable sigmoid(u) = 1/(1+exp(-u))
    p = zeros(size(u));
    idx = u >= 0;
    p(idx)  = 1 ./ (1 + exp(-u(idx)));
    eu      = exp(u(~idx));
    p(~idx) = eu ./ (1 + eu);
end