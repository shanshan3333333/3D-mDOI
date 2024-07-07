load pa_imgs
load bg_imgs
% use longest exposure time to show
boundingbox=object_extraction(imgs{3},80,300,0.2,true);
patch={};
loc={};
num=0;
%% delete the box edge of the phantom
for ii = 1:4
offset=80;
[imgs,bg_img]=extract_ROI(imgs,bg_imgs,boundingbox,offset);
%% hdr generation 
MaximumLimit=2^12;
expTimes=[50;150;300];
hdr = hdr_generation(imgs,expTimes,bg_img,MaximumLimit,1);
hdr=hdr-double(bg_img{1});
figure()
gray=mat2gray(imgs{3});
imshow(gray)
roi_range.u=-1;
roi_range.d=-1;
roi_range.l=-1;
roi_range.r=-1;
if roi_range.l==-1
    roi_range.l=1;
end
if roi_range.u==-1
    roi_range.u=1;
end
if roi_range.d==-1 && roi_range.r==-1
    [roi_range.d,roi_range.r]=size(imgs{3});
end
% show box
hold on
plot([roi_range.l roi_range.r], [roi_range.u roi_range.u],'r')
hold on
plot([roi_range.l roi_range.r], [roi_range.d roi_range.d],'r')
hold on
plot([roi_range.l roi_range.l], [roi_range.u roi_range.d],'r')
hold on
plot([roi_range.r roi_range.r], [roi_range.u roi_range.d],'r')
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
roi=resizing( roi,0.2,0.2*8);
[N.y,N.x]=size(roi);
figure()
gray=mat2gray(roi);
imshow(gray)
mask = localMaximum(roi,3,80,0.01);




% find nearest dots
[y,x]=find(mask~=0);
n_dot=length(x);
minDist=3;
% find_dist ->help yourself
% dist=round(find_dist(y,x,minDist));
dist=4;
% slice patchs containing one dot



for i=1:n_dot
    
    if y(i)+dist<=N.y && y(i)-dist>0 && x(i)+dist<=N.x && x(i)-dist>0
        num=num+1;
        patch{num}=roi(y(i)-dist:y(i)+dist,x(i)-dist:x(i)+dist);
        patch{num}=patch{num}/sum(sum(patch{num}));
        patch{num}(patch{num}==0) = min(min(patch{num}(patch{num}>0)));
%         patch{num} = log(patch{num});
        loc{num}=[y(i)-dist,x(i)-dist];
    else
        mask(y(i),x(i))=0;
    end
end
figure()
gray=logical(mask);
imshow(gray);

msg=strcat('the number of dots calculated in the roi : ',num2str(num));
disp(msg);
if ii==1
    load pb_imgs
elseif ii ==2 
    load pc_imgs
elseif ii ==3 
    load pd_imgs
else
    disp('error');
end
    
end

