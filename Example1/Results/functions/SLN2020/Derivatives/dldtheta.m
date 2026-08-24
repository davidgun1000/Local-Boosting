function dl_dtheta = dldtheta(sintheta,costheta)
m = size(costheta,1);
% %Using sparse matrices
% dl_dtheta = sparse(m^2,m*(m-1));
% for i = 2:m
%     for j = 1:m
%         if i==j %Derivative for diagonal elements in L
%             for k = 1:m
%                 if k == i-1
%                     dl_dtheta = dl_dtheta + sparse((j-1)*m+i,(i-1)*(m-1)+k,prod(sintheta(i,1:(i-2)),2)*costheta(i,i-1),m^2,m*(m-1));
%                 elseif k<i-1
%                     dl_dtheta = dl_dtheta + sparse((j-1)*m+i,(i-1)*(m-1)+k,prod(sintheta(i,1:(i-2)),2)*sintheta(i,i-1)*(costheta(i,k)/sintheta(i,k)),m^2,m*(m-1));
%                 end
%             end
%         elseif i>j %Derivative for off diagonal elements in L
%             for k = 1:m
%                 if k == j
%                     dl_dtheta = dl_dtheta + sparse((j-1)*m+i,(i-1)*(m-1)+k,-prod(sintheta(i,1:(j-1)),2)*sintheta(i,j),m^2,m*(m-1));
%                 elseif k<j
%                     dl_dtheta = dl_dtheta + sparse((j-1)*m+i,(i-1)*(m-1)+k, prod(sintheta(i,1:(j-1)),2)*costheta(i,j)*(costheta(i,k)/sintheta(i,k)),m^2,m*(m-1));
%                     dl_dtheta((j-1)*m+i,(i-1)*(m-1)+k) =  prod(sintheta(i,1:(j-1)),2)*costheta(i,j)*(costheta(i,k)/sintheta(i,k));
%                 end
%             end
%         end
%     end
% end

%Using non sparse matrix
dl_dtheta = zeros(m^2,m*(m-1));
for i = 2:m
    for j = 1:m
        if i==j %Derivative for diagonal elements in L
            for k = 1:m
                if k == i-1
                    dl_dtheta((j-1)*m+i,(i-1)*(m-1)+k) = prod(sintheta(i,1:(i-2)),2)*costheta(i,i-1);
                elseif k<i-1
                    dl_dtheta((j-1)*m+i,(i-1)*(m-1)+k) = prod(sintheta(i,1:(i-2)),2)*sintheta(i,i-1)*(costheta(i,k)/sintheta(i,k));
                end
            end
        elseif i>j %Derivative for off diagonal elements in L
            for k = 1:m
                if k == j
                    dl_dtheta((j-1)*m+i,(i-1)*(m-1)+k) = -prod(sintheta(i,1:(j-1)),2)*sintheta(i,j);
                elseif k<j
                    dl_dtheta((j-1)*m+i,(i-1)*(m-1)+k) =  prod(sintheta(i,1:(j-1)),2)*costheta(i,j)*(costheta(i,k)/sintheta(i,k));
                end
            end
        end
    end
end

ind = tril(reshape(1:((m-1)*m),m-1,m)',-1)';
inlast = ind(ind~=0);
dl_dtheta = sparse(dl_dtheta(:,inlast));