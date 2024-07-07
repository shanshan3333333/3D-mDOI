% used to read several images
% input:
%            file_loc-> location that you store the raw images
%            title   -> a cell containing different letter
%            group_size-> the max number appearing in image name
%            mode-> whether to do the average or not
% output:
%            imgs->     a averaged or sum-up image
% e.g.
% file_loc='C:\Users\cais\Downloads\phantom image\dark\';
% title={'a','s','d','f','g'};
% group_size=9 the maximum number of the group
% mode -> true: average false: sum up
function result=read_image(file_loc,title,group_size,mode)
    if ~islogical(mode)
       error('Error. \nInput mode must be a boolean, not a %s.',class(mode))
    end
    group=size(title);
    n_img=group(2)*group_size;
    img=cell(n_img,1);
    for i=1:group(2)
        for j=0:group_size
          name=strcat(file_loc,title(i),int2str(j),'.tif');
          img{(i-1)*10+(j+1)}=imread(name{1});
        end
    end
    disp('finish loading images');
    [h,w]=size(img{1});
    if mode==true
        result=uint16(zeros(h,w));
        for i=1:n_img
        result=result+img{i}/n_img;
        end
    else
        result=double(zeros(h,w));
        for i=1:n_img
            result=result+double(img{i});
        end
    end
    figure()
    gray=mat2gray(result);
    imshow(gray)
end