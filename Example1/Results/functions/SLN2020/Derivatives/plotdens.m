function plotdens(obj,varargin)
eps = 1e-10;
yl = icdf(obj,eps);
yu = icdf(obj,1-eps);
ygrid = [yl-1 yl:((yu-yl)/(500)):yu yu+1];
plot(ygrid,pdf(obj,ygrid),varargin{:});


