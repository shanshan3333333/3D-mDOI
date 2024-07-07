function param=combine_hv(H,V)
false_h = H ==0;
false_v = V ==0;
proper=~(false_h | false_v);
[Y,X]=size(H);
mask=ones(Y,X);
mask(proper)=2;
param =(H+V)./mask;
end

