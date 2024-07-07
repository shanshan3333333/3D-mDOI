%MAKEHDR    Create high dynamic range image.
%   'imgs'          cells containing images
%
%   'ExposureValues'    A vector of exposure values. unit can be ignored
%
%   'bg_imgs'           cell of bg imgs
%
%   'MaximumLimit'      A numeric scalar value in the range [0 255] that
%                       specifies the maximum "correctly exposed" value.
%                       For each low dynamic range image, pixels with
%                       larger values are considered overexposed and will
%                       not contribute to the final high dynamic range
%                       image.

% minimumlimit ????????????????????
%    'base index'       set base file            
function hdr = hdr_generation(imgs,expTimes,bg_imgs,MaximumLimit,base_index)
% check parameters
if length(imgs)~=length(expTimes)
    disp('error! the length of images does not equal to the length of the expTimes');
    return
elseif length(imgs)<=2
    disp('error! need at least two images to do the correction');
    return 
elseif base_index>length(expTimes) || base_index<1
    disp('error! index of basefile is invalid');
    return
end

% initialization
img_num=length(expTimes);
img_size=size(imgs{1});
someUnderExposed = false(img_size);
someOverExposed = false(img_size);
someProperlyExposed = false(img_size);
properlyExposedCount = zeros(img_size);
hdr=zeros(img_size);
% ????????
relExposure=expTimes./expTimes(base_index);

for i = 1:img_num
    ldr=imgs{i};
    MinimumLimit=bg_imgs{i};
    underExposed = ldr < MinimumLimit+5;
    someUnderExposed = someUnderExposed | underExposed;
    
    overExposed = ldr > MaximumLimit;
    someOverExposed = someOverExposed | overExposed;
    
    properlyExposed = ~(underExposed | overExposed);
    someProperlyExposed = someProperlyExposed | properlyExposed;
    % where properly exposed is true add one in preperly exposed count     
    properlyExposedCount(properlyExposed) = properlyExposedCount(properlyExposed) + 1;
    
    % Remove over- and under-exposed values.
    ldr(~properlyExposed) = 0;
    
    % Bring the intensity of the LDR image into a common HDR domain by
    % "normalizing" using the relative exposure, and then add it to the
    % accumulator.
    hdr = hdr + double(ldr ./ relExposure(i));
end

% Average the values in the accumulator by the number of LDR images that
% contributed to each pixel to produce the HDR radiance map.
hdr = hdr ./ max(properlyExposedCount, 1);

% For pixels that were completely over-exposed, assign the maximum
% value computed for the properly exposed pixels.
% ??????????????
hdr(someOverExposed & ~someUnderExposed & ~someProperlyExposed) = max(hdr(someProperlyExposed));

% For pixels that were completely under-exposed, assign the
% minimum value computed for the properly exposed pixels.
% ????????
hdr(someUnderExposed & ~someOverExposed & ~someProperlyExposed) = min(hdr(someProperlyExposed));

% For pixels that were sometimes underexposed, sometimes
% overexposed, and never properly exposed, use regionfill.
fillMask = someUnderExposed & someOverExposed & ~someProperlyExposed;
if any(fillMask(:))
    hdr = regionfill(hdr, fillMask);
end


