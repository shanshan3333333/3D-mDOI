% try to remove background
% use the assumption that the top left corner of the images is background
% should use the background dot with base images
% param: test window+offset
%        visible-> sample background patch and original images
% win=80;
% thres=0.1;
% off_set=300;
% visible=false;
function radius=soucelight_subtraction(img,visible)
% estimate the length of the dot
[H,W]=size(img);
mask = localMaximum(img,10,100,0.1);
[y,x]=find(mask~=0);
n_dot=length(x);
if length(x)<=2 
    disp('please use larger window size');
    return 
else
    msg=strcat('the number of dots in the images : ',num2str(n_dot));
    disp(msg);
end
if visible== true
    figure()
    gray=mat2gray(img);
    imshow(gray)
    figure()
    imshow(mask)
end

dist=round(sqrt((x(1)-x(2))^2+(y(1)-y(2))^2)/2);
num=0;
total=0;
for i=1:n_dot
    if y(i)+dist<H && y(i)-dist>0 && x(i)+dist<W && x(i)-dist>0
        patch=img(y(i)-dist:y(i)+dist,x(i)-dist:x(i)+dist);
        param=h_gaussian(patch ,false);
        total=total+0.5*(param(3)+param(5));
        num=num+1;
    end
end
radius=round(total/num);
end




