% function:  read_patternimgs
% aim:       get average images with different exposure times from several raw images 
%            save them to a specific location  
% input:
%            file_loc-> location that you store the raw images
%            save_loc-> location that you want to save the result
%            name    -> the name of the pattern 
%            bg      -> whether the images are background or not
% output:    None
% e.g. save pattern a with the name of 'pa_imgs.mat' and the variance
%      should be imgs
%      save background with the name of 'bg_imgs.mat' and the variance 
%      should be bg_imgs
function read_patternimgs(file_loc,save_loc,name,bg)
if ~islogical(bg)
   error('Error. \nInput bg must be a boolean, not a %s.',class(mode))
end
if bg==true
    name=strcat(name,'\');
    bg_name=strcat(file_loc,name);
    bg_imgs=read_groupimages(bg_name);
    save_name=strcat(save_loc,'bg_imgs.mat');
    save(save_name,'bg_imgs');
else
    p_name=strcat(name,'\');
    imgs_name=strcat(file_loc,p_name);
    imgs=read_groupimages(imgs_name);
    save_name=strcat(save_loc,name,'_imgs.mat');
    save(save_name,'imgs');
end
