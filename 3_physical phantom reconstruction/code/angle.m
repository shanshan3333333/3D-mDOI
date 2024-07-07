function degree=angle(y,x,c_y,c_x)
var=@(y,x)atan((y-c_y)/(x-c_x))*180/pi;
if y-c_y<=0
    if x-c_x>0
        degree=180+var(y,x);
    else
        degree=abs(var(y,x));
        
    end
else
    if x-c_x>=0
        degree=180+var(y,x);
    else
        degree=360+var(y,x);
    end    
end