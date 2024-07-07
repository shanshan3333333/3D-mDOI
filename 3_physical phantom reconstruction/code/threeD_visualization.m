function threeD_visualization(data)
[Nx,Ny,Nz]=size(data);
x_binsize   = 0.02; 	% size of each bin, eg. [cm] or [mm]
z_binsize   = 0.005;     % size of depth of the bin[cm]
x=1:Nx;
y=1:Ny;
x= x_binsize*(x-Nx/2);
y= x_binsize*(y-Ny/2);
% show
zslice = 1:10:Nz;
z = -z_binsize*zslice;
N=length(zslice);
Nrol=ceil(N/3);
figure()
set(gcf,'color','w');
min_var=zeros(1,N);
max_var=zeros(1,N);
for i=1:N
    var=data(:,:,zslice(i));
    min_var(i)=min(min(var));
    max_var(i)=max(max(var));
end
bottom=min(min_var);
top=max(max_var);

for i=1:N
    caxis manual
    caxis([bottom top]);
    subplot(Nrol,3,i);
    var=data(:,:,zslice(i));
    min_var(i)=min(min(var));
    max_var(i)=max(max(var));
    imagesc(x,y,var);
    title(num2str(z(i)))
end
caxis manual
caxis([bottom top]);
hp4 = get(subplot(Nrol,3,N),'Position');
colorbar('Position', [hp4(1)+hp4(3)+0.01  hp4(2)  0.05  hp4(3)])