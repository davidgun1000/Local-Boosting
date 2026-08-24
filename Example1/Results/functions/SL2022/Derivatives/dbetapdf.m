function deri = dbetapdf(x,par1,par2,pdfval)
if nargin ==3
    pdfval = betapdf(x,par1,par2);
end
deri = pdfval.*(((par2-1)./(1-x))+((par1-1)./(x)));