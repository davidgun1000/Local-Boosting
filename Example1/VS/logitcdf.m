function y=logitcdf(x,PhiMax)
% input x in (-inf,inf)
%global PhiMax
y=1./(1+exp(-x)); % transform in (0,1)
%y=PhiMax*y; % transform in (0,phimax)

