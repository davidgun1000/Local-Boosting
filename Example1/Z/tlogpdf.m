function [y] = tlogpdf(x, nu, mu, sigma)
%TLOGPDF  Log-pdf of the location-scale Student's t; optional score d/dx log f.
%   y = TLOGPDF(x, nu)                 % standard t (mu=0, sigma=1)
%   y = TLOGPDF(x, nu, mu, sigma)      % location-scale
%   [y, score] = TLOGPDF(...)          % also returns d/dx log f at x
%
%   - Vectorized over x (and supports implicit expansion with parameters).
%   - Requires nu > 0, sigma > 0.

    if nargin < 3 || isempty(mu),    mu = 0; end
    if nargin < 4 || isempty(sigma), sigma = 1; end

    % light checks
    if any(nu <= 0, 'all');    error('nu must be > 0.');    end
    if any(sigma <= 0, 'all'); error('sigma must be > 0.'); end

    z  = (x - mu) ./ sigma;
    c  = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log(nu*pi) - log(sigma);
    y  = c - ((nu+1)/2) .* log1p( (z.^2) ./ nu );   % stable log(1+·)

    if nargout > 1
        % score: d/dx log t_nu(x;mu,sigma) = - ((nu+1)(x-mu)) / (nu*sigma^2 + (x-mu).^2)
        score = - (nu + 1) .* (x - mu) ./ (nu .* sigma.^2 + (x - mu).^2);
    end
end