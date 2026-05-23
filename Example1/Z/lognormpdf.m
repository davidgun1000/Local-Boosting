function y = lognormpdf(x, mu, sigma)
%LOGNORMPDF  Log-pdf of a univariate Normal N(mu, sigma^2).
%   y = LOGNORMPDF(x)            % standard normal (mu=0, sigma=1)
%   y = LOGNORMPDF(x, mu, sigma) % location-scale
%
%   Vectorized over x. Requires sigma > 0.

    if nargin < 2 || isempty(mu),    mu = 0; end
    if nargin < 3 || isempty(sigma), sigma = 1; end
    if any(sigma <= 0, 'all')
        error('sigma must be > 0.');
    end

    z = (x - mu) ./ sigma;
    y = -0.5*log(2*pi) - log(sigma) - 0.5*(z.^2);
end