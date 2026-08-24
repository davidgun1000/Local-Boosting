function dXT_dkappa = dxT_dkappa(kappa)
[n,k] = size(kappa);
sinkappa = sin(kappa);
coskappa = cos(kappa);

%dx = zeros(k+1,k,n);
dXT_dkappa = sparse((k+1)*n,k*n);
for j = 1:k+1
    for l = 1:min(j,k)
        if j<(k+1)
            if j==l
                %                                     dx(j,l,:) =  -prod(sinkappa(:,1:l),2);
                dXT_dkappa = dXT_dkappa + sparse((k+1)*((1:n)-1)+j,(k)*((1:n)-1)+l,-prod(sinkappa(:,1:l),2),(k+1)*n,k*n) ;
            else
                %                                     dx(j,l,:) =  prod(coskappa(:,[j l]),2).*prod(sinkappa(:,[1:(l-1) (l+1):(j-1)]),2);
                dXT_dkappa = dXT_dkappa + sparse((k+1)*((1:n)-1)+j,(k)*((1:n)-1)+l,prod(coskappa(:,[j l]),2).*prod(sinkappa(:,[1:(l-1) (l+1):(j-1)]),2),(k+1)*n,k*n) ;
                
            end
        else
            if l==k
                %                                     dx(j,l,:) =  coskappa(:,l).*prod(sinkappa(:,1:(l-1)),2);
                dXT_dkappa = dXT_dkappa + sparse((k+1)*((1:n)-1)+j,(k)*((1:n)-1)+l,coskappa(:,l).*prod(sinkappa(:,1:(l-1)),2),(k+1)*n,k*n) ;
                
            else
                %                                     dx(j,l,:) =  coskappa(:,l).*prod(sinkappa(:,[1:(l-1) (l+1):end]),2);
                dXT_dkappa = dXT_dkappa + sparse((k+1)*((1:n)-1)+j,(k)*((1:n)-1)+l,coskappa(:,l).*prod(sinkappa(:,[1:(l-1) (l+1):end]),2),(k+1)*n,k*n) ;
            end
        end
    end
end


% dXT_dkappa = sparse((k+1)*n,k*n);
% [col,row] = meshgrid(1:(k),1:(k+1));
% for i = 1:n
%     colini = k*(i-1);
%     rowini = (k+1)*(i-1);
%     dxi_dkappa_i = dx(:,:,i);
%     dXT_dkappa = dXT_dkappa + sparse(rowini+row,colini+col,dxi_dkappa_i(:),(k+1)*n,k*n);  %Filling in as sparse block matrices
% end


   
