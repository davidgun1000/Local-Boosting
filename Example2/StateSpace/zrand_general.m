function z = zrand_general(sz, mu, s, a, b)
%ZRAND_GENERAL  Sample from Z (logit–Beta/EGB2) with general a,b.
%   z = ZRAND_GENERAL(sz, mu, s, a, b)
%   Z = mu + s * logit(P),   P ~ Beta(a,b),   logit(p)=log(p)-log(1-p)

    if nargin < 2 || isempty(mu), mu = 0; end
    if nargin < 3 || isempty(s),  s  = 1; end
    if nargin < 4 || isempty(a),  a  = 0.5; end
    if nargin < 5 || isempty(b),  b  = 0.5; end

    if isscalar(sz), sz = [sz, 1]; end
    assert(isscalar(mu) && isscalar(s) && isscalar(a) && isscalar(b), ...
        'mu, s, a, b must be scalars.');
    assert(s > 0 && a > 0 && b > 0, 'Require s>0, a>0, b>0.');

    %--- sample P ~ Beta(a,b) ---
    if exist('betarnd','file')
        p = betarnd(a, b, sz);
    elseif exist('gamrnd','file')
        g1 = gamrnd(a, 1, sz);
        g2 = gamrnd(b, 1, sz);
        p  = g1 ./ (g1 + g2);
    else
        error(['Need BETARND or GAMRND (Stats Toolbox). ', ...
               'Otherwise plug in your own Beta/Gamma sampler.']);
    end

    %--- stable logit transform (avoid 0 or 1 exactly) ---
    p = min(max(p, eps), 1 - eps);    % clip to (eps, 1-eps)
    z0 = log(p) - log1p(-p);          % log(1 - p) computed stably

    z = mu + s * z0;
end