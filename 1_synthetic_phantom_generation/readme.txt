We are using the code written by Scott Prahl to perform Monte Carlo simulations. Detailed tuning of the phantom parameter generation and Monte Carlo simulation can be found at https://omlc.org/software/mc/mcxyz/index.html.

Task 1: Simulate the synthetic dataset for testing
First, we run generate_phantom.m to produce _H.mci and _T.bin files, which initiate the next step of the Monte Carlo simulation. Then, we run batch_process.py, which automatically calls the compiled mcxyz.exec for the MAC system. The important outputs of this process are Ryz.bin and Rd.dat, which represent the reflectance data from Diffuse Optical Imaging. The convert_ryx2mat.m file then transfers the Ryz.bin files to .mat files, so they can be used for testing the 3d-mdoi code later on.

Task 2: Pre-compute the 3D photon distribution map for use in reconstruction
First, we run generate_pdf.m to produce _H.mci and _T.bin files. These files should contain the traits defined for the lookup map. If you run batch_process.py and call the compiled test.exec in the last line of the code for the MAC system, you will generate the pre-computed 3D photon distribution map for reconstruction. The outputs of this process are COUNT.bin and PDF.bin. Then, run combine_pdf.m to convert these results into pdf.mat and weight.mat, which are essential for the 3d-mdoi reconstruction.

We also include the source code for mcxyz.c and test.c as a reference. If you are using other computing system, please compile the .c file to exe first. Small adjustments to the filenames may be needed to match the different codes.

The test.exec requires a significant amount of memory space to store the 3D trajectories of the simulated photons. The system might become stuck if less space is allocated than needed. In this case, please reduce the Nbin value in the phantom settings to decrease memory usage.