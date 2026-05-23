function y=logitinverse(x,PhiMax)
% input x in (0,phimax)

%x=x/PhiMax; % transform in (0,1)
y=log(x./(1-x)); % transform to (-inf,inf)
