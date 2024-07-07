% function localMaximum
% find 2d local maximum and can handle saturated dots
% input:
% image-> TEST IMAGE
% minDist-> the min distance between the two dots
% minMaxvalue-> the min value of the peak of the dot
% ratio-> the threshold for selecting dot 
% OUTPUT: MASK of local maximum
function I = localMaximum(image,minDist,minMaxvalue,ratio)
max_value=max(max(image));
threshold=max_value*ratio;
if threshold < minMaxvalue
    threshold=minMaxvalue;
end
% handle saturated spot 
% assuming that the pixels are connected when saturated
[h,w]=size(image);
I= zeros(h,w);
I1=zeros(h,w);
a=find(image==max_value);
I1(a) = 1;
cc = bwconncomp(I1);
labeled = labelmatrix(cc);
for i=1:1:cc.NumObjects
    mask=zeros(h,w);
    a=find(labeled==i);
    mask(a) = 1;
    prop=regionprops(mask,'Centroid');
    center=round(prop.Centroid);
    I(center(2),center(1))=max_value;
end

% find peak point
[x,y]=find(image>threshold & image<max_value);
for k=1:1:length(x)
    flag=1;
    for i =-1:1:1
        for j=-1:1:1
            a=i+x(k);
            b=j+y(k);
            if(i~=0||j~=0)
                if((a>0)&&(a<=h)&&(b>0)&&(b<=w))
                    if (image(x(k),y(k))-image(a,b)<=0)
                        flag=0;
                        break;
                    end
                end
            end
        end        
        if flag==0
            break;
        end
    end
    if flag==1
        I(x(k),y(k))=image(x(k),y(k));
    end
end



% clean nearest spot
% if the value are the same, keep the first one
I1=I;
[x,y]=find(I~=0);
for i=1:1:length(x)
    a=find(x>x(i)-minDist & x<x(i)+minDist);
    for j=1:1:length(a)
        if i<a(j)
            if (y(a(j))>y(i)-minDist && y(a(j))<y(i)+minDist)
                if I1(x(i),y(i))< I1(x(a(j)),y(a(j)))
                    I(x(i),y(i))=0;
                else
                    I(x(a(j)),y(a(j)))=0;
                end
            end
        end
    end
end
end

