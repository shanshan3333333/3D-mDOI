function x=light_param(p,r,g,A,off)
lb = [0,50];
ub = [1,500];
% x0 value is the standard tissue 
x0 = [0.01,100];
% p is distance 
% r is value
fixedset = logical([0 0 1 1 1]);
fixedvals = [g,A,off];
opts = optimset('Display','off');
[x,~,~,~] = lsqcurvefit(@(x,data) RTEwrapper(x,data,fixedset,fixedvals),x0,p,r,lb,ub,opts);


