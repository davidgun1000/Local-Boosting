function [pdf] = t_pdf(x,mu,sigma,nu)

    z = (x - mu) ./ sigma;
    c = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log(nu*pi) - log(sigma);
    y = c - ((nu+1)/2) .* log1p( (z.^2) ./ nu );
    pdf = exp(y);

end