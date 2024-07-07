function new_img_lp=rotation(img,degree)
    [m, n, z] = size(img);
    new_m = ceil(abs(m*cosd(degree)) + abs(n*sind(degree)));
    new_n = ceil(abs(n*cosd(degree)) + abs(m*sind(degree)));
    new_img_lp=zeros(new_m,new_n,z);
    % reverse mapping matrices
    rm1 = [1 0 0; 0 -1 0; -0.5*new_n 0.5*new_m 1];
    rm2 = [cosd(degree) sind(degree) 0; -sind(degree) cosd(degree) 0; 0 0 1];
    rm3 = [1 0 0; 0 -1 0; 0.5*n 0.5*m 1];

    for i = 1:new_n
       for j = 1:new_m
           % rotated image's coordinates to no-rotation image's coordinates
          old_coordinate = [i j 1]*rm1*rm2*rm3;
          col = old_coordinate(1);
          row = old_coordinate(2);
          % prevent border overflow 
          if row < 1 || col < 1 || row > m || col > n
              new_img_lp(j, i,:) = 0;
          else
              % bilinear interpolation
              left = floor(col);
              right = ceil(col);
              top = floor(row);
              bottom = ceil(row);

              a = col - left;
              b = row - top;
              new_img_lp(j, i, :) = (1-a)*(1-b)*img(top, left, :) + a*(1-b)*img(top, right, :) + ...
                  (1-a)*b*img(bottom, left, :) + a*b*img(bottom, right, :);
          end
       end
    end
    clip_m=ceil((new_m-m)/2);
    clip_n=ceil((new_n-n)/2);
    flag_m=mod(new_m-m,2);
    flag_n=mod(new_n-n,2);
    if clip_m~=0 
        if flag_m==0
            new_img_lp=new_img_lp(clip_m:new_m-clip_m-1,:,:);
        else
            new_img_lp=new_img_lp(clip_m:new_m-clip_m,:,:);
        end
    end
    if clip_n~=0
        if flag_n==0
            new_img_lp=new_img_lp(:,clip_n:new_n-clip_n-1,:);
        else
            new_img_lp=new_img_lp(:,clip_n:new_n-clip_n,:);
        end
    end
end