function [imgs,bg_imgs]=extract_ROI(imgs,bg_imgs,boundingbox,offset)

% adopt the boundingbox value to the images with different exposure time
imgs{1}=imgs{1}(boundingbox.u+offset:boundingbox.d-offset,boundingbox.l+offset:boundingbox.r-offset);
imgs{2}=imgs{2}(boundingbox.u+offset:boundingbox.d-offset,boundingbox.l+offset:boundingbox.r-offset);
imgs{3}=imgs{3}(boundingbox.u+offset:boundingbox.d-offset,boundingbox.l+offset:boundingbox.r-offset);
% % bg roi extraction
bg_imgs{1}=bg_imgs{1}(boundingbox.u+offset:boundingbox.d-offset,boundingbox.l+offset:boundingbox.r-offset);
bg_imgs{2}=bg_imgs{2}(boundingbox.u+offset:boundingbox.d-offset,boundingbox.l+offset:boundingbox.r-offset);
bg_imgs{3}=bg_imgs{3}(boundingbox.u+offset:boundingbox.d-offset,boundingbox.l+offset:boundingbox.r-offset);
end