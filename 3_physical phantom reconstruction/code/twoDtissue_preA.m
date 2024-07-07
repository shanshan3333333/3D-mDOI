function F=twoDtissue_preA(Fxy,win_size,Am,off)
bin_size   = 0.02; %cm
[Y,X]=size(Fxy);
s_x=X/2;
s_y=Y/2;
mua=zeros(Y,X);
mus=zeros(Y,X);
% set fitting param
g=0.9;
% vertical
dist=@(x,y) sqrt((x-s_x).^2+(y-s_y).^2);
for y=win_size+1:1:Y-win_size
    for x=1:X
        p_x=ones(1,2*win_size+1)*x;
        p_y=y-win_size:y+win_size;
        %adding small value to prevent divide zeros
        p=dist(p_y,p_x)*bin_size;
        r=Fxy(y-win_size:y+win_size,x);
        r=transpose(r);
        param=light_param(p,r,g,Am,off);
        mua(y,x)=param(1);
        mus(y,x)=param(2);
    end
end

V.mua=mua;
V.mus=mus;

mua=zeros(Y,X);
mus=zeros(Y,X);
for y=1:Y
    for x=win_size+1:1:X-win_size
        p_y=ones(1,2*win_size+1)*y;
        p_x=x-win_size:x+win_size;
        p=dist(p_y,p_x)*bin_size;
        r=Fxy(y,x-win_size:x+win_size);
        param=light_param(p,r,g,Am,off);
        mua(y,x)=param(1);
        mus(y,x)=param(2);

    end
end
H.mua=mua;
H.mus=mus;

F.mua=combine_hv(H.mua,V.mua);
F.mus=combine_hv(H.mus,V.mus);
end



