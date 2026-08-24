function kappa = tkappa2kappa(tkappa)
[~,p] = size(tkappa);
if p>0
epsi=1d-6*0; %MS edit
kappa = zeros(size(tkappa));
kappa(:,1:end-1) = normcdf(tkappa(:,1:end-1))*(pi-2*epsi)+epsi; %MS edit
kappa(:,end) = normcdf(tkappa(:,end))*(2*pi-2*epsi)+epsi; %MS edit
else
   kappa = tkappa;
end


