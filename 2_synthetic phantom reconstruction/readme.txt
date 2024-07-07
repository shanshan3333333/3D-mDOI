Checklist:

  Setup for compute_3dmdoi.py: The path activated for compute_3dmdoi.py must include dermis_pdf.mat and dermis_weight.mat. If correction_mua.npz and correction_mus.npz are not present in the current directory, please generate them before testing other sample cases.

  Simulated Data Location: Simulated data are stored in the folder named simulated_data, which contains three test cases: pure, shallow, and pigment. The pure case is used for correction generation. The shallow case simulates a phantom with pigment penetration up to 1mm (refer to main draft, Figure 2, Column B), while the pigment case simulates a phantom with pigment penetration up to 3mm (refer to main draft, Figure 2, Column C).

  Environment Setup: To run compute_3dmdoi.py successfully, set up the environment using the requirements.txt file in the folder. Then, modify the code at line 585 by assigning save_tag as "pure", "pigment", or "shallow" to select the test case.

  Running compute_3dmdoi.py: When you run compute_3dmdoi.py, it will:

   (1)Load the lookup map for banana shape generation.
   (2)Read raw images for different patterns.
   (3)Automatically calculate the bounding box of the object (phantom) from the background.
   (4)Find local maxima and collect patches of dots.
   (5)Fit each dot and obtain mua and mus.
   (6)Combine the 3D results from different patterns.
   (7)Save the output mua and mus in the current folder.
   (8)Visualize the 2D slice results for the 3D mua and mus matrix. 

  Conversion to TIFFs: converter3D.py is used to convert the output mua.npz file into several TIFFs, enabling further transfer to IMS files via the Imaris file converter and view via Imaris viewer .(For more information, please refer to https://imaris.oxinst.com/learning/view/article/importing-data-into-imaris and https://imaris.oxinst.com/imaris-viewer)

  Quantitative Analysis: The result_data folder contains IMS files of 3D-mDOI, FEM, and Ground-truth for both visualization and quantitative analysis. Running quantitative_measurement.py directly will reproduce the case 2,3 of quantitative measurement in Table 1 of the main draft.