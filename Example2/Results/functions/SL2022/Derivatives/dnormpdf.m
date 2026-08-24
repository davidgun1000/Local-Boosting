function deri = dnormpdf(x,mu,sigma,pdfval)
if nargin ==3
    pdfval = normpdf(x,mu,sigma);
end
deri = pdfval.*(-(x-mu));


