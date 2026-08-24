function dkappadtkappa = dkappa_dtkappa(tkappa)
epsi=1d-6*0; %MS edit
nkappa = numel(tkappa);
dkappadtkappa = zeros(size(tkappa));
dkappadtkappa(:,1:end-1) = normpdf(tkappa(:,1:end-1))*(pi-2*epsi); %MS edit
dkappadtkappa(:,end) = normpdf(tkappa(:,end))*(2*pi-2*epsi); %MS edit
dkappadtkappa = dkappadtkappa';
dkappadtkappa = sparse(1:nkappa,1:nkappa,dkappadtkappa(:));
