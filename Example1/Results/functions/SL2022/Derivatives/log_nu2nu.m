function [nu,dnu_dlog_nu] = log_nu2nu(log_nu)
ulim = 100;
llim = 2;
explognu = expma(log_nu);
nu = (ulim*explognu+llim)/(explognu+1);
dnu_dlog_nu = explognu*((ulim*(explognu+1)-(ulim*explognu+llim))/((explognu+1)^2));