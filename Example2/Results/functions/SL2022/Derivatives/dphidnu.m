function val = dphidnu(w,B,z,d,eps,dw)
% w=F^{-1}_W(u;nu) and dw= (d/d.nu) F^{-1}_W(u;nu)
Bz_deps = B*z + d.*eps;
val = 0.5*w^(-0.5)*dw*Bz_deps;