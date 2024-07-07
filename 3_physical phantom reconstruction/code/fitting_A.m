function Am=fitting_A(Fxy,R)
fit = @(x,R) x(1)*R+x(2);
x0 = [sum(sum(Fxy)),min(min(Fxy))];
opts = optimset('Display','off');
Am = lsqcurvefit(fit,x0,R,Fxy,[],[],opts);