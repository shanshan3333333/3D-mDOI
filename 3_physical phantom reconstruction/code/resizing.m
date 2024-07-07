function result=resizing(img,bin_length,x_bin)
ratio=round(x_bin/bin_length);
[h,w]=size(img);
H=floor(h/ratio);
W=floor(w/ratio);
result=zeros(H,W);
for hi = 1:H
    for wi =1:W
        result(hi,wi)=sum(sum(double(img((hi-1)*ratio+1:hi*ratio,(wi-1)*ratio+1:wi*ratio))));
    end
end
