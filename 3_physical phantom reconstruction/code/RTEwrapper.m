function pred = RTEwrapper(xvary,data,fixedset,fixedvals)
  x = zeros(size(fixedset));
  x(fixedset) = fixedvals;
  x(~fixedset) = xvary;
  pred = RTEFunction(x,data);
end