function [logp, dlogp] = logpdf_deriv_mix2t(x, w1, nu1, mu1, sigma1, w2, nu2, mu2, sigma2)
% LOGPDF_DERIV_MIX2T  Log-density and d/dx log-density for a 2-component t-mixture.
%
%   [logp, dlogp] = LOGPDF_DERIV_MIX2T(x, w1, nu1, mu1, sigma1, w2, nu2, mu2, sigma2)
%
% Inputs:
%   x       : array of evaluation points (vectorized)
%   w1,w2   : nonnegative mixture weights (will be normalized internally)
%   nu1,nu2 : degrees of freedom (>0)
%   mu1,mu2 : locations (real)
%   sigma1,sigma2 : scales (>0)
%
% Outputs:
%   logp  : log mixture density at x
%   dlogp : derivative w.r.t. x of the log mixture density at x
%
% Notes:
%   - Uses log-sum-exp for numerical stability.
%   - d/dx log p(x) = r1(x)*s1(x) + r2(x)*s2(x),
%     where s_k is the score of the k-th t component and r_k are responsibilities.

%     arguments
%         x
%         w1   (1,1) double {mustBeNonnegative}
%         nu1  (1,1) double {mustBePositive}
%         mu1  (1,1) double
%         sigma1 (1,1) double {mustBePositive}
%         w2   (1,1) double {mustBeNonnegative}
%         nu2  (1,1) double {mustBePositive}
%         mu2  (1,1) double
%         sigma2 (1,1) double {mustBePositive}
%     end

    % --- normalize weights to sum to 1 (robust to unnormalized inputs) ---
    ws = w1 + w2;
    if ws == 0
        error('At least one mixture weight must be positive.');
    end
    w1 = w1 / ws;
    w2 = w2 / ws;

    % --- component log-densities ---
    logt1 = tlogpdf_loc(x, nu1, mu1, sigma1);
    logt2 = tlogpdf_loc(x, nu2, mu2, sigma2);

    % --- log mixture via log-sum-exp ---
    a = log(w1) + logt1;
    b = log(w2) + logt2;
    m = max(a, b);
    logp = m + log( exp(a - m) + exp(b - m) );

    % --- component scores s_k(x) = d/dx log t_nu(x; mu, sigma) ---
    s1 = - (nu1 + 1) .* (x - mu1) ./ (nu1 * sigma1.^2 + (x - mu1).^2);
    s2 = - (nu2 + 1) .* (x - mu2) ./ (nu2 * sigma2.^2 + (x - mu2).^2);

    % --- responsibilities r_k(x) = w_k t_k(x) / p(x) computed stably ---
    r1 = exp( (log(w1) + logt1) - logp );
    r2 = exp( (log(w2) + logt2) - logp );

    % --- derivative of log mixture density ---
    dlogp = r1 .* s1 + r2 .* s2;
end

% --------- helper: log-pdf of location-scale Student-t (stable) ----------
function y = tlogpdf_loc(x, nu, mu, sigma)
    z = (x - mu) ./ sigma;
    c = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log(nu*pi) - log(sigma);
    y = c - ((nu+1)/2) .* my_log1p( (z.^2) ./ nu );  % log(1 + z.^2/nu) stably
end

% --------- helper: stable log1p with fallback ----------
function y = my_log1p(u)
    if exist('log1p','builtin') || exist('log1p','file')
        y = log1p(u);
    else
        y = log(1 + u);
    end
end