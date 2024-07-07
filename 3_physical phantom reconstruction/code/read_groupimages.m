%read group of images from s/m/l file independently
%the format of image name shoule be like '[a letter from title][a number from 0-9].tif'
% input:
%            file_loc-> location that you store the raw images
% output:
%            imgs->     a cell containing 3 images with different exposure
%            time{short, median, long}
function imgs=read_groupimages(file_loc)
title={'a','s','d','f','g'};
loc=strcat(file_loc,'s\');
imgs{1}=read_image(loc,title,9,true);
loc=strcat(file_loc,'m\');
imgs{2}=read_image(loc,title,9,true);
loc=strcat(file_loc,'l\');
imgs{3}=read_image(loc,title,9,true);

