function [y]=vec(X)

[m n]=size(X);
y=reshape(X,m*n,1);
