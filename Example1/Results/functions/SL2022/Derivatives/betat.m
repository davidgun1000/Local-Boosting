function phi = betat(theta,par1,par2)
phi = norminv(betacdf(normcdf(theta),par1,par2));
end


