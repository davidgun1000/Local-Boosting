function [vecL,L,indvech] = vechL2vecL_horz(vechL,n,q)
indvech = [];
for  k = 1:n
   indvech = [indvech k:q:(k*(q))];
end
vecL = zeros(n*q,1);
vecL(indvech) = vechL;
L = reshape(vecL,n,q);
