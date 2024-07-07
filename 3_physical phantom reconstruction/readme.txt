Checklist:

The activated path for the Matlab CODE folder should contain pdf_map.mat, w_map.mat, and correction.mat.
If you run the main file from the top of the script, you need the raw image data from the OPTIMAP_data folder, containing the following files:
	pa_imgs.mat
	pb_imgs.mat
	pc_imgs.mat
	pd_imgs.mat
	bg_imgs.mat
Each input file in the OPTIMAP_data folder contains 3 cells, representing the collected reemitted diffuse data with exposures of 50, 150, and 300 ms, respectively.

The code in the code folder comprises the 3D-mDOI reconstruction pipeline, which has been tested with MATLAB R2017a. Using a more recent version of MATLAB may require corresponding modifications to the functions.

When you run the main file, it will:

(1) Load the lookup map for banana shape generation.
(2) Read raw images and generate average images for different patterns and backgrounds.
(3) Automatically calculate the bounding box of the object (phantom) from the background.
(4) Perform the pattern calculation at the selected ROI.
(5) Combine the 3D results of different patterns.
(6) Visualize the 3D results for mua and mus.

Inside the function of pattern_calculation in step 4, the code will:

(1) Extract the object (phantom) from a cell of images and a cell of backgrounds with different exposure times.
(2) Generate an HDR image using the image with the shortest exposure time as the base image.
(3) Manually cut the ROI region.
(4) Calculate the radius of the spots' center (not in use now).
(5) Find local maxima and collect patches of dots.
(6) Fit each dot and obtain mua and mus (function twoDtissue).
(7) Use threeD_generation to reconstruct the 3D light coefficient map.
(8) Save the output mua and mus in phantom_result.mat.

After obtaining the phantom_result.mat, the code for further analytical processes is stored in analysis.py. Ensure that phantom_result.mat, fem_result.npz, and bg_Ref.mat are stored in the result_data folder for the proper running of the code. The script will plot the reference row and the quantitative analysis in paper figure 3 and supplementary figure 4.

The ims_file folder in result_data has the ims files of 3D-mDOI, FEM and Ground-truth for visualization.