% Inside the function of pattern_calculation, the code will...
%    1)extract object(phantom) from cell of images and cell of backgrounds with different exposure time
%    2)generate hdr image using the image with shortest exposure time as base image
%    3)manually cut roi region 
%    4)calculate radius of the spots' center[not in use now]
%    5)find local maximum and collect patches of dots
%    6)fit each dot and obtain mua and mus(function twoDtissue)  
%    7)use threeD_generation to reconstruct 3D light coefficient map 
% Input 
%     imgs-> a cell of images[different exposure time]
%     boundingbox ->bounding box gained from object extraction
%     roi_range-> select a certain roi to calculate
%     pdf_map&w_map-> 3D look-up table
% output
%     mua,mus-> 3d parameters
%     w_mua,w_mus -> pdf for 3D mua and mus
%     loc-> location of patches
%     N-> size of selected roi
%     twoD_param-> 2d light coefficients
function [mua,w_mua,mus,w_mus,loc,N,twoD_param]=pattern_calculation_preA(imgs,bg_imgs,boundingbox,roi_range,pdf_map,w_map)
%% delete the box edge of the phantom
offset=80;
[imgs,bg_imgs]=extract_ROI(imgs,bg_imgs,boundingbox,offset);
%% hdr generation 
MaximumLimit=2^12;
expTimes=[50;150;300];
hdr = hdr_generation(imgs,expTimes,bg_imgs,MaximumLimit,1);
hdr=hdr-double(bg_imgs{1});
% figure()
% gray=mat2gray(imgs{3});
% imshow(gray)
if roi_range.l==-1
    roi_range.l=1;
end
if roi_range.u==-1
    roi_range.u=1;
end
if roi_range.d==-1 && roi_range.r==-1
    [roi_range.d,roi_range.r]=size(imgs{3});
end
% % show box
% hold on
% plot([roi_range.l roi_range.r], [roi_range.u roi_range.u],'r')
% hold on
% plot([roi_range.l roi_range.r], [roi_range.d roi_range.d],'r')
% hold on
% plot([roi_range.l roi_range.l], [roi_range.u roi_range.d],'r')
% hold on
% plot([roi_range.r roi_range.r], [roi_range.u roi_range.d],'r')
%% center extraction 
% [H,W]=size(pa_s);
% offset=round(H/2);
% ref_img=pa_s(offset-100:offset+100,boundingbox.l-300:boundingbox.l-100);
% ref_bg=bg_s(offset-100:offset+100,boundingbox.l-300:boundingbox.l-100);
% radius=soucelight_subtraction(ref_img-ref_bg,true);
% msg=strcat('radius for central light source :',num2str(radius));
% disp(msg);

%% roi maximum point calculation
disp('select a patch from hdr result to test');
% uniform roi
% roi=double(hdr(270:360,30:150));
% mix roi
% roi=double(hdr(390:480,450:570));
roi=double(hdr(roi_range.u:roi_range.d,roi_range.l:roi_range.r));

[N.y,N.x]=size(roi);
figure()
gray=mat2gray(roi);
imshow(gray)

mask = localMaximum(roi,20,80,0.01);

% find nearest dots
[y,x]=find(mask~=0);
n_dot=length(x);
% minDist=3;
% find_dist ->help yourself
% dist=round(find_dist(y,x,minDist));
dist = 17;

% slice patchs containing one dot
num=0;
patch={};
loc={};
for i=1:n_dot
    
    if y(i)+dist<=N.y && y(i)-dist>0 && x(i)+dist<=N.x && x(i)-dist>0
        num=num+1;
        patch{num}=roi(y(i)-dist:y(i)+dist,x(i)-dist:x(i)+dist);
        patch{num}=patch{num}/sum(sum(patch{num}));
        patch{num}(patch{num}==0) = min(min(patch{num}(patch{num}>0)));
        patch{num} = log(patch{num});
        loc{num}=[y(i)-dist,x(i)-dist];
    else
        mask(y(i),x(i))=0;
    end
end
% figure()
% gray=logical(mask);
% imshow(gray);

msg=strcat('the number of dots calculated in the roi : ',num2str(num));
disp(msg);
%% fitting A first

% [Y,X]=size(patch{1});
% R=simulation_R(Y,X);
% % set fitting param
% if num>10
%     fit_num = 10;
% else
%     fit_num = num;
% end
% for i=1:fit_num
%     Am{i}=fitting_A(patch{i},R);
% end
% Am=cell2mat(Am);
% Am=reshape(Am,2,[]);
% Am=mean(Am');
A=0.24;
off = -7.04;
msg=strcat('average Am :',num2str(A));
disp(msg)
%% fitting
load('correction.mat')
win_size=7;
for i=1:num
    twoD_param{i}=twoDtissue_preA(patch{i},win_size,A,off);
    [mua{i},w_mua{i}]=threeD_generation(twoD_param{i}.mua,pdf_map,w_map,correct_mua);
    [mus{i},w_mus{i}]=threeD_generation(twoD_param{i}.mus,pdf_map,w_map,correct_mus);
    if rem(i,50)==0
        msg='finish number of dot:';
        msg=strcat(msg,num2str(i));
        disp(msg);
    end
end
disp('finish one pattern')


