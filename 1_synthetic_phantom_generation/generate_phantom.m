% generate light dots around pigemnt
% main testing sampel to use
% author shanshan cai
close all
Nbins=64;
Nsource=1;
x_binsize   = 0.05; 	%cm 3.5cm

dx = x_binsize;
Nx = Nbins;
x  = ([1:Nx]'-Nx/2)*dx;


step=4;
x_offset=-15:step:15;
y_offset=-15:step:15;
index=1;
ss={};


for yi=1:length(y_offset)
    for xi=1:length(x_offset)
        xs=x_offset(xi)+Nbins/2*[1:Nsource];
        ys=y_offset(yi)+Nbins/2*[1:Nsource];
        Tyx=zeros(Nx,Nx);
        for i =1:Nsource
            for j=1:Nsource
                Tyx(ys(j)-2:ys(j)+2,xs(i)-2:xs(i)+2)=1;
                ss{index}=strcat("ss",int2str(ys(j)),"_",int2str(xs(i)));
                name=strcat("/output/ss",int2str(ys(j)),"_",int2str(xs(i)))
                makePigmentphantom((ys(j)-Nx/2)*dx,(xs(i)-Nx/2)*dx,Nbins,name)
                index=index+1;
            end
        end
%         figure()
%         set(gcf,'color','w');
%         imagesc(x,x,Tyx)
    end
end

    
