function twoD_visualization(data)
[Nx,Ny]=size(data{1});
x_binsize   = 0.02; 	% size of each bin, eg. [cm] or [mm]

x=1:Nx;
y=1:Ny;
x= x_binsize*(x-Nx/2);
y= x_binsize*(y-Ny/2);
% show
N=length(data);
Nrol=ceil(N/3);
figure()
for i=1:N
    subplot(Nrol,3,i);
    imagesc(x,y,data{i});
end
hp4 = get(subplot(Nrol,3,N),'Position');
colorbar('Position', [hp4(1)+hp4(3)+0.01  hp4(2)  0.05  hp4(3)])