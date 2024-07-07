function R=simulation_R(Y,X)
bin_size   = 0.02; %cm
s_x=X/2;
s_y=Y/2;
p=zeros(Y,X);
% vertical
dist=@(x,y) sqrt((x-s_x).^2+(y-s_y).^2)*bin_size;
for y = 1:Y
    for x=1:X
        p(y,x)=dist(y,x);
    end
end
standard_tissue=[0.01,100,0.9,1,0];
R=RTEFunction(standard_tissue,p);