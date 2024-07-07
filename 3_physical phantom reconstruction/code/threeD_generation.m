% clc;
% clear;
function [result,weight]=threeD_generation(oneD,pdf_map,w_map,cc)
Nbins  = 62;    	% # of bins in each dimension of cube 

[By,Bx]=size(w_map);
[Ny,Nx]=size(oneD);
Nz=Nbins; 
threeD=double(zeros(Ny,Nx,Nz));
W=double(zeros(Ny,Nx,Nz));

c_x=floor(Nx/2);
c_y=floor(Ny/2);
y_offset=c_y-By;
x_offset=c_x-Bx;
u_3d=max(1,1+y_offset);
d_3d=min(Ny,Ny-y_offset);
l_3d=max(1,1+x_offset);
r_3d=min(Ny,Ny-x_offset);
u_pdf=max(1,1-y_offset);
d_pdf=min(Nbins,Nbins+y_offset+1);
l_pdf=max(1,1-x_offset);
r_pdf=min(Nbins,Nbins+x_offset+1);


for y=1:Ny
    dy=c_y-y;
    sign_y=sign(dy);
    dy=abs(dy);    
    if dy>=By
        continue
    end
    for x=1:Nx
        dx=c_x-x;
        sign_x=sign(dx);
        dx=abs(dx);
        if dx>=Bx
            continue
        end
        map=pdf_map{By-dy,Bx-dx};
        if isempty(map)
            continue
        end
        param=oneD(y,x);
        if param==0
            continue
        end
        w=w_map{By-dy,Bx-dx};
        if sign_x<0
            map=threeDflip(map,'lr');
        end
        if sign_y<0
            map=threeDflip(map,'ud');
        end
        map=map(u_pdf:d_pdf,l_pdf:r_pdf,:);
        W(u_3d:d_3d,l_3d:r_3d,:)=W(u_3d:d_3d,l_3d:r_3d,:)+map*w;
        threeD(u_3d:d_3d,l_3d:r_3d,:)=threeD(u_3d:d_3d,l_3d:r_3d,:)+map*w*param; 
    end  
end
weight=W;
mask=W==0;
W(mask)=1;
result=threeD./W;
% todo correct weight

result=result./cc;




