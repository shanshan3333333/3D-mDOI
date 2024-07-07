% % MAIN is the script to calculate 3D light coefficient based on 2D phantom images
% % 
% % Run the main file and it will....
% %    1)load look up map for banana shape generation
% %    2)read raw images and generate average images for different patterns and backgournd
% %    3)automatically calculate boundingbox of object(phantom) from the background
% %    4)do the pattern calculation at the selected ROI
% %    5)combine the 3D result of different patterns
% %    6)visualize the 3D results for mua and mus
% % 
% % written by Shanshan Cai 
% % 05/29/2018
% %% 
clc;
close all;
clear;
% % %% load data matrix
load pdf_map.mat
load w_map.mat
% % %% read img data
% % file_loc='C:\Users\cais\Downloads\polarizer\';
% % save_loc='C:\Users\cais\Downloads\phantom image\polarizer\';
% % % bg
% % read_patternimgs(file_loc,save_loc,'bg',true);
% % % pattern
% % read_patternimgs(file_loc,save_loc,'pa',false);
% % read_patternimgs(file_loc,save_loc,'pb',false);
% % read_patternimgs(file_loc,save_loc,'pc',false);
% % read_patternimgs(file_loc,save_loc,'pd',false);
% 
% 
% %% object extraction 
load pa_imgs
load bg_imgs
% use longest exposure time to show
boundingbox=object_extraction(imgs{3},80,300,0.2,true);

% %% pattern calculation
% % choose a area
% mix_roi.u=350;
% mix_roi.d=450;
% mix_roi.l=100;
% mix_roi.r=250;
mix_roi.u=-1;
mix_roi.d=-1;
mix_roi.l=-1;
mix_roi.r=-1;
tic
[mua{1},w_mua{1},mus{1},w_mus{1},loc{1},N,twoD_param{1}]=pattern_calculation_preA(imgs,bg_imgs,boundingbox,mix_roi,pdf_map,w_map);
load pb_imgs
[mua{2},w_mua{2},mus{2},w_mus{2},loc{2},~,twoD_param{2}]=pattern_calculation_preA(imgs,bg_imgs,boundingbox,mix_roi,pdf_map,w_map);
load pc_imgs
[mua{3},w_mua{3},mus{3},w_mus{3},loc{3},~,twoD_param{3}]=pattern_calculation_preA(imgs,bg_imgs,boundingbox,mix_roi,pdf_map,w_map);
load pd_imgs
[mua{4},w_mua{4},mus{4},w_mus{4},loc{4},~,twoD_param{4}]=pattern_calculation_preA(imgs,bg_imgs,boundingbox,mix_roi,pdf_map,w_map);
% % % save fitting data for further usage
param.mua=mua;
param.mus=mus;
param.loc=loc;
param.w_mua=w_mua;
param.w_mus=w_mus;
param.twoD_param=twoD_param;
param.N=N;
% save mix_param_c.mat param
%% combination

c_mua=combine_cell(mua);
c_mus=combine_cell(mus);
c_wmua=combine_cell(w_mua);
c_wmus=combine_cell(w_mus);
c_loc=combine_cell(loc);
multi_mua_0=multicombination(c_mua,c_wmua,c_loc,N,-1);
multi_mus_0=multicombination(c_mus,c_wmus,c_loc,N,-1);
toc
threeD_visualization(multi_mua_0);
threeD_visualization(multi_mus_0);
% disp('finish')
multi_mua_1=multicombination(c_mua,c_wmua,c_loc,N,1e-2);
multi_mus_1=multicombination(c_mus,c_wmus,c_loc,N,1e-2);
threeD_visualization(multi_mua_1);
threeD_visualization(multi_mus_1);
disp('finish')
multi_mua_2=multicombination(c_mua,c_wmua,c_loc,N,1e-3);
multi_mus_2=multicombination(c_mus,c_wmus,c_loc,N,1e-3);
threeD_visualization(multi_mua_2);
threeD_visualization(multi_mus_2);
disp('finish')
multi_mua_3=multicombination(c_mua,c_wmua,c_loc,N,1e-1);
multi_mus_3=multicombination(c_mus,c_wmus,c_loc,N,1e-1);
threeD_visualization(multi_mua_3);
threeD_visualization(multi_mus_3);
disp('finish')

% % visualization
% load mus
% load mua
threeD_visualization(multi_mua);
threeD_visualization(multi_mus);
% % visualize two D light coefficents
% [mua_2D,mus_2D]=twoD_format(twoD_param)
% twoD_visualization(mua_2D)
% twoD_visualization(mus_2D)