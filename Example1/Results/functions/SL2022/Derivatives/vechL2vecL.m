function [vecL,L] = vechL2vecL(vechL,n,q)
% ind = zeros(q*(q-1)/2,1);
% s = 0;
% for  k = 2:q
%    for j = 2:k
%        s = s+1;
%        ind(s) = (k-1)*n+(j-1);
%    end
% end
%intT = (1:(q*n))';
%indvech = setdiff(intT,ind);

indvech = zeros(n*q-(q-1)*q*0.5,1);
for  k = 1:q
    inivalind = (k-1)*n-(k-2)*(k-1)*0.5+1;
    endvalind = k*n-(k-1)*k*0.5;
    indvech(inivalind:endvalind) = (((k-1)*n+k):(n*k))';
end

if numel(vechL(:))~=(n*q-(q-1)*q*0.5)
    error('Incorrect number of elements for the creation of vec') 
end

vecL = zeros(n*q,1);
vecL(indvech) = vechL;
L = reshape(vecL,n,q);