
function [result,reliability]=multicombination(param,weight,loc,N,thres)
Nz=62;
result=zeros(N.y,N.x,Nz);
W=zeros(N.y,N.x,Nz);
n_spot=length(loc);
for i =1:n_spot
    l=loc{i};
    [By,Bx,~]=size(param{i});
    W(l(1):l(1)+By-1,l(2):l(2)+Bx-1,:)=W(l(1):l(1)+By-1,l(2):l(2)+Bx-1,:)...
        +weight{i};
end
ratio=max(max(max(W)));
reliability=W./ratio;
% threeD_visualization(log(W));
mask=W==0;
W(mask)=1;

for i =1:n_spot
    l=loc{i};
    [By,Bx,~]=size(param{i});
    result(l(1):l(1)+By-1,l(2):l(2)+Bx-1,:)=result(l(1):l(1)+By-1,l(2):l(2)+Bx-1,:)...
        +param{i}.*weight{i};
end
result=result./W;

if thres>0
    mask=reliability<=thres;
    result(mask)=0;
end