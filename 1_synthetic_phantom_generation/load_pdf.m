function [F,n_count,Ny,Nx,Nz]=load_pdf(myname)
format compact
commandwindow

SAVEPICSON = 1;
if SAVEPICSON
    sz = 10; fz = 7; fz2 = 5; % to use savepic.m
else
    sz = 20; fz = 20; fz2 = 7; % for screen display
end

%%%% USER CHOICES <---------- you must specify -----
nm = 680;
%%%%


disp(sprintf('------ mcxyz %s -------',myname))

% Load header file
filename = sprintf('%s_H.mci',myname);
disp(['loading ' filename])
fid = fopen(filename, 'r');
A = fscanf(fid,'%f',[1 Inf])';
fclose(fid);

%% parameters
time_min = A(1);
Nx = A(2);
Ny = A(3);
Nz = A(4);
dx = A(5);
dy = A(6);
dz = A(7);
mcflag = A(8);
launchflag = A(9);
boundaryflag = A(10);
xs = A(11);
ys = A(12);
zs = A(13);
xfocus = A(14);
yfocus = A(15);
zfocus = A(16);
ux0 = A(17);
uy0 = A(18);
uz0 = A(19);
radius = A(20);
waist = A(21);
Nt = A(22);
j = 22;
n_loc=Nx/2;
for i=1:Nt
    j=j+1;
    muav(i,1) = A(j);
    j=j+1;
    musv(i,1) = A(j);
    j=j+1;
    gv(i,1) = A(j);
end

reportHmci(myname)

%% Load Banana Shape F(,y,x,z) 
filename = sprintf('%s_PDF.bin',myname);
disp(['loading ' filename])
tic
    fid = fopen(filename, 'rb');
    [Data count] = fread(fid, n_loc*Ny*Nx*Nz, 'double');
    fclose(fid);
toc
F = reshape(Data,Ny,Nx,Nz,n_loc); % F(y,x,z)


%% Load N_count
filename = sprintf('%s_COUNT.bin',myname);
disp(['loading ' filename])
tic
    fid = fopen(filename, 'rb');
    [n_count count] = fread(fid, n_loc*Ny*Nx*Nz, 'double');
    fclose(fid);
toc



end