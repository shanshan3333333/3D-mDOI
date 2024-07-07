function flip_map=threeDflip(map,direction)
[Ny,Nx,Nz]=size(map);
flip_map=zeros([Ny,Nx,Nz]);
if strcmp(direction,'lr')
    for z=1:Nz
        flip_map(:,:,z)=fliplr(map(:,:,z));
    end
elseif strcmp(direction,'ud')
    for z=1:Nz
        flip_map(:,:,z)=flipud(map(:,:,z));
    end
else
    disp('wrong input direction');
    return 
end
end