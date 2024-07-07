Fs={};
ns_count={};
[Fs{1},ns_count{1},~,~,~]=load_pdf("ss_1");
[Fs{2},ns_count{2},~,~,~]=load_pdf("ss_2");
[Fs{3},ns_count{3},~,~,~]=load_pdf("ss_3");
[Fs{4},ns_count{4},~,~,~]=load_pdf("ss_4");
[Fs{5},ns_count{5},Ny,Nx,Nz]=load_pdf("ss_5");

F = (Fs{1} + Fs{2} + Fs{3} + Fs{4} + Fs{5})/5;
n_count = (ns_count{1} +ns_count{2} + ns_count{3} + ns_count{4} + ns_count{5})/5;
% mark output region
dx=0.005;
dz=0.02;
x = ([1:Nx]-Nx/2-1/2)*dx;
y = ([1:Ny]-Ny/2-1/2)*dx;
z = ([1:Nz]-1/2)*dz;
ux = [2:Nx-1];
uy = [2:Ny-1];
uz = [2:Nz-1];
zmin = min(z);
zmax = max(z);
zdiff = zmax-zmin;
xmin = min(x);
xmax = max(x);
xdiff = xmax-xmin;
myname = 'phantom_PDF';

% %% look banana value
% for loc = 1:28
%     Fzx = reshape(F(Ny/2,:,:,loc),Nx,Nz)'; % in z,x plane through source
%     % normalize it 
%     Fzx = Fzx/max(max(Fzx));
% 
%     figure();clf
%     set(gcf,'color','w');
%     imagesc(x,z,Fzx)
%     hold on
%     text(max(x)*1.2,min(z)-0.04*max(z),'PDF_','fontsize',fz)
%     colorbar
%     set(gca,'fontsize',sz)
%     xlabel('x [cm]')
%     ylabel('z [cm]')
%     name = strcat('PDF__',myname);
%     title(name,'fontweight','normal','fontsize',30)
%     colormap(makec2f)
%     axis equal image
%     %axis([min(x) max(x) min(z) max(z)])
%     text(min(x)-0.2*max(x),min(z)-0.08*max(z),sprintf('runtime = %0.1f min',time_min),...
%         'fontsize',fz2)
% end
% 
% disp('done')


%% load 
F=F(2:200,1:199,:,:);%resize

pdf_map = cell(100,100);
w_map = cell(100,100);
all_weight=0;
for yi=1:100
    for xi =1:100
        dist = sqrt((yi-100)^2 + (xi-100)^2);
        if dist>99
            continue;
        end
        dist = 100 - dist;
        floor_dist = floor(dist);
        ceil_dist = ceil(dist);
        degree=angle(yi,xi,100,100);
        if floor_dist==ceil_dist
            img=F(:,:,:,floor_dist);
            weight=n_count(floor_dist);
        else
            img_1=F(:,:,:,floor_dist);
            img_2=F(:,:,:,ceil_dist);
            weight_1 = abs(ceil_dist-dist);
            weight_2 = abs(floor_dist-dist);
            img = img_1*weight_1 +img_2*weight_2;
            w_1 = n_count(floor_dist);
            w_2 = n_count(ceil_dist);
            weight =w_1*weight_1 +w_2*weight_2 ;
        end
        if isnan(degree)
            pdf_map{yi,xi}= img/max(max(max(img))); 
        else
            new_img_lp=rotation(img,degree);
            pdf_map{yi,xi}= new_img_lp/max(max(max(new_img_lp)));
        end
        w_map{yi,xi} = weight;
        all_weight =all_weight+weight;

    end
end
for yi=1:100
    for xi =1:100
        w_map{yi,xi}=w_map{yi,xi}/all_weight;
    end
end
% phantoms->0.05*2
% phantom1cmN -> 0.05 icm

save('phantom_pdf.mat', 'pdf_map', '-v7.3')
save phantom_weight.mat w_map