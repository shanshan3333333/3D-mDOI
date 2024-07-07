function makePigmentphantom(xs,ys,Nbins,myname)
%   tissue types and the beam being launched.
%
%   Uses
%       makeTissueList.m
%
%   To use, 
%       1. Prepare makeTissueList.m so that it contains the tissue
%   types desired.
%       2. Specify the USER CHOICES.
%       2. Run this program, maketissue.m.
%
%   Note: mcxyz.c can use optical properties in cm^-1 or mm^-1 or m^-1,
%       if the bin size (binsize) is specified in cm or mm or m,
%       respectively.
%
%  Steven L. Jacques. updated Aug 21, 2014.
%       


format compact
close all

%%% USER CHOICES %%%%%%%% <-------- You must set these parameters ------
SAVEON      = 1;        % 1 = save myname_T.bin, myname_H.mci 
                        % 0 = don't save. Just check the program.

% myname      = 'plain';% name for files: myname_T.bin, myname_H.mci  
time_min    = 5;      	% time duration of the simulation [min] <----- run time -----
% nm          = 532;   	% desired wavelength of simulation green
nm         =680;      %red 
% nm         =450;      %blue 



% Nbins       = 62;    	% # of bins in each dimension of cube 
x_binsize   = 0.05; 	% size of each bin, eg. [cm] or [mm]
z_binsize   = 0.02;     % size of depth of the bin[cm] 

% Set Monte Carlo launch flags
mcflag      = 0;     	% launch: 0 = uniform beam, 1 = Gaussian, 2 = isotropic pt. 
                        % 3 = rectangular beam (use xfocus,yfocus for x,y halfwidths)
launchflag  = 0;        % 0 = let mcxyz.c calculate launch trajectory
                        % 1 = manually set launch vector.
boundaryflag = 2;       % 0 = no boundaries, 1 = escape at boundaries
                        % 2 = escape at surface only. No x, y, bottom z
                        % boundaries

% Sets position of source
% xs          = 0;      	% x of source
% ys          = 0;        % y of source
zs          = 0.0;  	% z of source

% Set position of focus, so mcxyz can calculate launch trajectory
xfocus      = xs;        % set x,position of focus
yfocus      = ys;        % set y,position of focus
zfocus      = inf;    	% set z,position of focus (=inf for collimated beam)

% only used if mcflag == 0 or 1 or 3 (not 2=isotropic pt.)
radius      = 0.0010;   % 1/e radius of beam at tissue surface
waist       = 0.0010;  	% 1/e radius of beam at focus

% only used if launchflag == 1 (manually set launch trajectory):
ux0         = 0.7;      % trajectory projected onto x axis
uy0         = 0.4;      % trajectory projected onto y axis
uz0         = sqrt(1 - ux0^2 - uy0^2); % such that ux^2 + uy^2 + uz^2 = 1
%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%% 
% Prepare Monte Carlo 
%%%

% Create tissue properties
tissue = makeTissueList(nm); % also --> global tissue(1:Nt).s
Nt = length(tissue);
for i=1:Nt
    muav(i)  = tissue(i).mua;
    musv(i)  = tissue(i).mus;
    gv(i)    = tissue(i).g;
end

% Specify Monte Carlo parameters    
Nx = Nbins;
Ny = Nbins;
Nz = 50;
dx = x_binsize;
dy = x_binsize;
dz = z_binsize;
x  = ([1:Nx]'-Nx/2)*dx;
y  = ([1:Ny]'-Ny/2)*dy;
z  = [1:Nz]'*dz;
zmin = min(z);
zmax = max(z);
xmin = min(x);
xmax = max(x);

if isinf(zfocus), zfocus = 1e12; end

%%%%%%
% CREATE TISSUE STRUCTURE T(y,x,z)
%   Create T(y,x,z) by specifying a tissue type (an integer)
%   for each voxel in T.
%
%   Note: one need not use every tissue type in the tissue list.
%   The tissue list is a library of possible tissue types.

T = double(zeros(Ny,Nx,Nz)); 

% T=T+4 % fill background with skin (dermis)
% all the the phantom is pigment
T = T + 4;      

zsurf = 0.000;  % position of air/skin surface

for iz=1:Nz % for every depth z(iz)

% %     air
%     if iz<=round(zsurf/dz)
%         T(:,:,iz) = 1; 
%     end
% 
%     % epidermis (60 um thick)
%     if iz>round(zsurf/dz) & iz<=round((zsurf+0.0060)/dz)
%         T(:,:,iz) = 5; 
%     end
    
    

    %pigment  @ xc, zc, radius, oriented along y axis

    xc      = 0;            % [cm], center of blood vessel
    yc      = 0;            % [cm], center of blood vessel
    pigmentdiameter  = 0.2;      	% blood vessel radius [cm]
%     shallow 0.1pigment 0.3 surface0.05 deepest0.5 mid between 0.1~0.3
    pigment_depth = 0.3;
    zc      = 0;     	% [cm], center of blood vessel
    z_offset = 0;
    
%         for ix=1:Nx
%         for iy=1:Ny
%             xd = x(ix) - xc;	% vessel, x distance from vessel center
%             yd = y(iy) - yc;	% vessel, x distance from vessel center
%             zd = z(iz) - zc;   	% vessel, z distance from vessel center                
%             r  = sqrt((xd/pigmentdiameter)^2 +(yd/pigmentdiameter)^2);	% r from vessel center
%             if (r<=1 && zd + z_offset>0.1 && z_offset+zd<0.5)     	% if r is within vessel
%                 T(iy,ix,iz) = 10; % blood
%             end
%         end
%         end 
%     half football
    
    for ix=1:Nx
        for iy=1:Ny
            xd = x(ix) - xc;	% vessel, x distance from vessel center
            yd = y(iy) - yc;	% vessel, x distance from vessel center
            zd = z(iz) - zc;   	% vessel, z distance from vessel center                
            r  = sqrt((xd/pigmentdiameter)^2 + (zd/pigment_depth)^2+(yd/pigmentdiameter)^2);	% r from vessel center
            if (r<=1)     	% if r is within vessel
                T(iy,ix,iz) = 10; % blood
            end
        end
    end %ix

    
end % iz

% tag = 'surface_gt';
% for iz = 1:Nz
%     file = T(:,:,iz);
%     file=imbinarize(file,5).*255;
%     name=strcat(tag,int2str(iz),'.tiff');
%     imwrite(file,name,'tiff');
% end
%%
if SAVEON
    tic
    % convert T to linear array of integer values, v(i)i = 0;
    v = uint8(reshape(T,Ny*Nx*Nz,1));

    %% WRITE FILES
    % Write myname_H.mci file
    %   which contains the Monte Carlo simulation parameters
    %   and specifies the tissue optical properties for each tissue type.
    commandwindow
    disp(sprintf('--------create %s --------',myname))
    
    filename = sprintf('%s%s_H.mci',pwd,myname);
    fid = fopen(filename,'w');
        % run parameters
        fprintf(fid,'%0.2f\n',time_min);
        fprintf(fid,'%d\n'   ,Nx);
        fprintf(fid,'%d\n'   ,Ny);
        fprintf(fid,'%d\n'   ,Nz);
        fprintf(fid,'%0.4f\n',dx);
        fprintf(fid,'%0.4f\n',dy);
        fprintf(fid,'%0.4f\n',dz);
        % launch parameters
        fprintf(fid,'%d\n'   ,mcflag);
        fprintf(fid,'%d\n'   ,launchflag);
        fprintf(fid,'%d\n'   ,boundaryflag);
        fprintf(fid,'%0.4f\n',xs);
        fprintf(fid,'%0.4f\n',ys);
        fprintf(fid,'%0.4f\n',zs);
        fprintf(fid,'%0.4f\n',xfocus);
        fprintf(fid,'%0.4f\n',yfocus);
        fprintf(fid,'%0.4f\n',zfocus);
        fprintf(fid,'%0.4f\n',ux0); % if manually setting ux,uy,uz
        fprintf(fid,'%0.4f\n',uy0);
        fprintf(fid,'%0.4f\n',uz0);
        fprintf(fid,'%0.4f\n',radius);
        fprintf(fid,'%0.4f\n',waist);
        fprintf(fid,'%0.4f\n',zsurf);
        % tissue optical properties
        fprintf(fid,'%d\n',Nt);
        for i=1:Nt
            fprintf(fid,'%0.4f\n',muav(i));
            fprintf(fid,'%0.4f\n',musv(i));
            fprintf(fid,'%0.4f\n',gv(i));
        end
    fclose(fid);

    %% write myname_T.bin file
    filename = sprintf('%s%s_T.bin',pwd,myname);
    disp(['create ' filename])
    fid = fopen(filename,'wb');
    fwrite(fid,v,'uint8');
    fclose(fid);

    toc
end % SAVEON


%% Look at structure of Tzx at iy=Ny/2
Txzy = shiftdim(T,1);   % Tyxz --> Txzy
Tzx  = Txzy(:,:,Ny/2)'; % Tzx

%%
figure(1); clf
sz = 12;  fz = 10; 
set(gcf,'color','w');
imagesc(x,z,Tzx,[1 Nt])
hold on
set(gca,'fontsize',sz)
xlabel('x [cm]')
ylabel('z [cm]')
title('standard tissue')
colorbar
cmap = makecmap(Nt);
colormap(cmap)
set(colorbar,'fontsize',1)
% label colorbar
zdiff = zmax-zmin;
%%%

for i=1:Nt
    yy = (Nt-i)/(Nt-1)*Nz*dz;
    text(max(x)*1.2,yy, tissue(i).name,'fontsize',fz)
end

text(xmax,zmin - zdiff*0.06, 'Tissue types','fontsize',fz)
axis equal image
axis([xmin xmax zmin zmax])

%%% draw launch
N = 20; % # of beam rays drawn
switch mcflag
    case 0 % uniform
        for i=0:N
            plot((-radius + 2*radius*i/N)*[1 1],[zs max(z)],'r-')
        end

    case 1 % Gaussian
        for i=0:N
            plot([(-radius + 2*radius*i/N) xfocus],[zs zfocus],'r-')
        end

    case 2 % iso-point
        for i=1:N
            th = (i-1)/19*2*pi;
            xx = Nx/2*cos(th) + xs;
            zz = Nx/2*sin(th) + zs;
            plot([xs xx],[zs zz],'r-')
        end
        
    case 3 % rectangle
        zz = max(z);
        for i=1:N
            xx = -radius + 2*radius*i/20;
            plot([xx xx],[zs zz],'r-')
        end
end




%% Look at structure of Tyx at iz=zc
 %Tyxz-->Tyx
%%
figure(2)
sz = 12;  fz = 10; 
set(gcf,'color','w');
slice_num=9;
resolution = Nz / slice_num;
for i=1:slice_num
    index=round(resolution*(i-1)+1);
    Tyx  = T(:,:,index);
    subplot(3,3,i);
    imagesc(y,x,Tyx,[1 Nt])
    name=strcat('depth of the tissue ',num2str(z(index))," mm");
    title(name)
    xlabel('x [cm]')
    ylabel('y [cm]')

end

hold on
set(gca,'fontsize',sz)
suptitle('standard tissue')
% colorbar
cmap = makecmap(Nt);
colormap(cmap)
% set(colorbar,'fontsize',1)
% % label colorbar
% xdiff = xmax-xmin;
% %%%
% 
% for i=1:Nt
%     yy = (Nt-i)/(Nt-1)*Nx*dx;
%     text(max(x)*1.2,yy-Nx/2*dx, tissue(i).name,'fontsize',fz)
% end
% 
% text(xmax,xmin - xdiff*0.06, 'Tissue types','fontsize',fz)
% axis equal image
% axis([xmin xmax xmin xmax])

%% draw launch
% N = 20; % # of beam rays drawn
% radius=radius*5;
% if  mcflag==0 || mcflag ==1
%     for i=0:N
%         plot((-radius + ys + 2*radius*i/N)*[1 1],[-radius+xs xs+radius],'r-')
%     end
% end

