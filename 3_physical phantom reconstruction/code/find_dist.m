function dist=find_dist(y,x,minDist)
flag=0;
for i=1:1:length(x)
    a=find(x>=x(i)-minDist & x<x(i)+minDist);
    if length(a)>1
        yy=y(a);
        yy=sort(yy);
        for j=1:1:length(a)
            if yy(j)~=y(i)
                dist=abs(yy(j)-y(i))/2;
                flag=1;
                break
            end
        end
    end
    if flag~=0
        break
    end
end