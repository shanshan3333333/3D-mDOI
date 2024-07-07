% try to remove background
% use the assumption that the top left corner of the images is background
% should use the images having longer exposure time
% input:
%        img-> input image 
%        window&offset-> find the place for the background area
%        thres -> set threshold of the bg average value
%        visible-> true to view sample background patch and original images
% win=80;
% thres=0.1;
% off_set=300;
% visible=false;
function boundingbox=object_extraction(img,win,offset,thres,visible)
[H,W]=size(img);
patch=img(offset:win+offset,offset:win+offset);

% estimate the length of the dot
mask = localMaximum(patch,10,500,0.9);
n_dot=find(mask~=0);
if length(n_dot)<=1 
    disp('please use larger window size');
    return
end
if visible== true
    figure()
    gray=mat2gray(img);
    imshow(gray)
    figure()
    gray=mat2gray(patch);
    imshow(gray);
    figure()
    imshow(mask)
end
avg_value=sum(sum(patch))*(1+thres);
mask=img(1:H-win,1:W-win);
[H,W]=size(mask);
for h =1:win:H
    for w = 1:win:W
        patch=img(h:h+win,w:w+win);
        value=sum(sum(patch));
        if value<avg_value
            mask(h:h+win,w:w+win)=0;
        end
    end
end

% detect edge
for h=1:1:H
    value=sum(mask(h,:));
    if value>0
        boundingbox.u=h;
        break;
    end
end
for h=H:-1:1
    value=sum(mask(h,:));
    if value>0
        boundingbox.d=h;
        break;
    end
end
for w=W:-1:1
    value=sum(mask(:,w));
    if value>0
        boundingbox.r=w;
        break;
    end
end
for w=1:1:W
    value=sum(mask(:,w));
    if value>0
        boundingbox.l=w;
        break;
    end
end

end




