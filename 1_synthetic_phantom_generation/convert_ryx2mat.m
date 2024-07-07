% main file genertate pattern map
close all


% make sure the current path is 
disp(['Current folder is now: ', pwd]);
% Generate a path string that includes all subfolders under the main folder
allSubFolders = genpath(pwd);

% Add all subfolders to the MATLAB search path
addpath(allSubFolders);


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
pattern_num=1;
savefile="./output/";
ux = [2:Nx-1]';       
for yi=1:length(y_offset)
    for xi=1:length(x_offset)
        xs=x_offset(xi)+Nbins/2*[1:Nsource];
        ys=y_offset(yi)+Nbins/2*[1:Nsource];
        Tyx=zeros(Nx,Nx);
        Ryx=zeros(Nx,Nx);
        for i =1:Nsource
            Tyx(ys,xs(i))=1;
            for j=1:Nsource
                ss{index}=strcat("ss",int2str(ys(j)),"_",int2str(xs(i)));
                name=strcat("ss",int2str(ys(j)),"_",int2str(xs(i)),"_Ryx.bin");
                disp(['loading ' name])
                fid = fopen(name, 'rb');
                [Data count] = fread(fid, Nx*Nx, 'float');
                fclose(fid);
                r=reshape(Data,Nx,Nx);
                U = r(ux,ux);
%                 figure()
%                 imagesc(x(ux),y(ux),U',[0 mean(U(:))*2])
                Ryx = Ryx+r;
                index=index+1;
                
            end
        end


        savename=strcat(savefile,"phantom_",int2str(pattern_num),'.mat')
        save(savename,"U")
        pattern_num=pattern_num+1;

%         figure()
%         set(gcf,'color','w');
%         imagesc(x,x,Tyx)
    end
end

    

