function tkappa = kappa2tkappa(kappa)
[~,p] = size(kappa);
if p>0
    epsi=1d-6; %MS edit
    tkappa = zeros(size(kappa));
    tkappa(:,1:end-1) = norminv((kappa(:,1:end-1)-epsi)/(pi-2*epsi)); %MS edit
    tkappa(:,end) = norminv((kappa(:,end)-epsi)/(2*pi-2*epsi)); %MS edit
else
    tkappa = kappa;
end
