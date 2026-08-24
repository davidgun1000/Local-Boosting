function [logp] = logmvnpdf2(B,z,d,eps)
p = size(B,2);
q = size(B,1);

Bz_deps = B*z + d.*eps;
DBz_deps = bsxfun(@times,Bz_deps,1./d.^2);

Dinv2B = bsxfun(@times,B,1./d.^2);

Half1 = DBz_deps;
Half2 = Dinv2B/(speye(p) + B'*Dinv2B)*B'*DBz_deps;

Blogdet = logdet(speye(p) + bsxfun(@times,B, 1./(d.^2))'*B) + sum(log((d.^2)));
logp = -( q/2*log(2*pi) + 1/2*Blogdet + 1/2*Bz_deps'*(Half1-Half2));