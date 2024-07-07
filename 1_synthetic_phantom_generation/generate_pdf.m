% compute pdf for new setting
close all
Nbins=21*2;
Nsource=1;
x_binsize   = 0.05; 	%cm 7.5cm

dx = x_binsize;
Nx = Nbins;
x  = ([1:Nx]'-Nx/2)*dx;

compute_copy = 5;
index=1;
ss={};

for i=1:compute_copy
        xs=Nbins/2;
        ys=Nbins/2;
        Tyx=zeros(Nx,Nx);
        name=strcat("/output/ss",int2str(i));
        makePigmentphantom(0,0,Nbins,name)
end
