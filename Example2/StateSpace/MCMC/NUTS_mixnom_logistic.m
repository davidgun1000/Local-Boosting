function [beta,alpha_ave,nfevals,logp,grad,depth]=NUTS_mixnom_logistic(f,epsilon,beta0,logp0,grad_beta0,max_tree_depth,inv_covmat,covmat)

global nfevals
nfevals=0;
d=length(beta0);
r0=(mvnrnd(zeros(1,d),inv_covmat))';
joint=logp0-0.5*(r0'*covmat*r0);
logu=joint-exprnd(1);
betaminus=beta0;
betaplus=beta0;
rminus=r0;
rplus=r0;
gradminus=grad_beta0;
gradplus=grad_beta0;
depth=0;

beta=beta0;
grad=grad_beta0;
logp=logp0;

n=1;
stop=false;

while ~stop
   dir=2*(rand()<0.5)-1;
   if dir == -1
      [betaminus,rminus,gradminus,~,~,~,betaprime,gradprime,logpprime,nprime,stopprime,alpha,nalpha]=...
          build_tree(betaminus,rminus,gradminus,logu,dir,depth,epsilon,f,joint,covmat);
   else
      [~,~,~,betaplus,rplus,gradplus,betaprime,gradprime,logpprime,nprime,stopprime,alpha,nalpha]=...
          build_tree(betaplus,rplus,gradplus,logu,dir,depth,epsilon,f,joint,covmat);
   end
   
   if (~stopprime && (rand<nprime/n))
       beta=betaprime;
       logp=logpprime;
       grad=gradprime;
   end
   
   %update number of valid points
   n=n+nprime;
   stop=stopprime || stop_criterion(betaminus,betaplus,rminus,rplus);
   %increment depth
   depth=depth+1;
   if depth>max_tree_depth
       disp('the current NUTS iteration reached the maximum tree depth');
       break
   end
    
end

alpha_ave=alpha/nalpha;
%alpha_ave=alpha;
end

function criterion=stop_criterion(thetaminus,thetaplus,rminus,rplus)
    thetavec=thetaplus-thetaminus;
    criterion=(thetavec'*rminus<0) || (thetavec'*rplus<0);
end

function [thetaprime,rprime,gradprime,logpprime]=leapfrog(theta,r,grad,epsilon,f,covmat)
    %keyboard
    rprime=r+0.5*epsilon*grad;
    thetaprime=theta+epsilon*covmat*rprime;
    
    
    [logpprime,gradprime]=f(thetaprime);
    
    rprime=rprime+0.5*epsilon*gradprime;
    global nfevals;
    nfevals=nfevals+1;
end

function [thetaminus,rminus,gradminus,thetaplus,rplus,gradplus,thetaprime,gradprime,logpprime,nprime,stopprime,alphaprime,nalphaprime]=...
    build_tree(theta,r,grad,logu,dir,depth,epsilon,f,joint0,covmat)
    %(Bminus,rminus,gradminus,logu,dir,depth,epsilon,f,joint,covmat)
    %keyboard
    %(betaminus,rminus,gradminus,logu,dir,depth,epsilon,f,joint,covmat,ctraj)
    if depth==0
        [thetaprime,rprime,gradprime,logpprime]=leapfrog(theta,r,grad,dir*epsilon,f,covmat);
        joint=logpprime-0.5*(rprime'*covmat*rprime);
        nprime=logu<joint;
        stopprime=logu-1000>=joint;
        thetaminus=thetaprime;
        thetaplus=thetaprime;
        rminus=rprime;
        rplus=rprime;
        gradminus=gradprime;
        gradplus=gradprime;
        %compute acceptance probability
        alphaprime=exp(logpprime-0.5*(rprime'*covmat*rprime)-joint0);
        if isnan(alphaprime)
            alphaprime=0;
        else
            alphaprime=min(1,alphaprime);
        end
        nalphaprime=1;
    
    else
        [thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, stopprime, alphaprime, nalphaprime] = ...
                build_tree(theta, r, grad, logu, dir, depth-1, epsilon, f,joint0,covmat);
        
        if ~stopprime
           if dir==-1
               [thetaminus,rminus,gradminus,~,~,~,thetaprime2,gradprime2,logpprime2,nprime2,stopprime2,alphaprime2,nalphaprime2]=...
                   build_tree(thetaminus,rminus,gradminus,logu,dir,depth-1,epsilon,f,joint0,covmat);
           else
               [~,~,~,thetaplus,rplus,gradplus,thetaprime2,gradprime2,logpprime2,nprime2,stopprime2,alphaprime2,nalphaprime2]=...
                   build_tree(thetaplus,rplus,gradplus,logu,dir,depth-1,epsilon,f,joint0,covmat);
           end
           
           if (rand()<nprime2/(nprime+nprime2))
               thetaprime=thetaprime2;
               gradprime=gradprime2;
               logpprime=logpprime2;
           end
           nprime=nprime+nprime2;
           stopprime=stopprime || stopprime2 || stop_criterion(thetaminus,thetaplus,rminus,rplus);
           alphaprime=alphaprime+alphaprime2;
           %alphaprime=max(alphaprime,alphaprime2);
           nalphaprime=nalphaprime+nalphaprime2;  
        end
    end
end