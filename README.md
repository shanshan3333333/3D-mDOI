# 3D-mDOI DEMO Instruction

## Task 1: Synthetic Phantom Generation
**Objective:** Generate a synthetic dataset for testing, including the creation of 3D trajectories of simulated photons.

### Steps:
1. Run `generate_phantom.m` to produce `_H.mci` and `_T.bin` files for initiating Monte Carlo simulations.
2. Use `batch_process.py` to call `mcxyz.exec` for MAC, generating reflectance data essential for Diffuse Optical Imaging.
3. Convert reflectance data to `.mat` files with `convert_ryx2mat.m` for further testing.

### Environment Requirements:
- **MATLAB:** MATLAB R2017a for `.m` files execution.
- **Python:** Python 3.6 environment with dependencies for `batch_process.py`.
- **Memory:** Sufficient memory allocation for `mcxyz.exec` to store 3D photon trajectories.

---

## Task 2: Synthetic Phantom Reconstruction
**Objective:** Conduct synthetic phantom reconstruction utilizing simulated data to create accurate 3D models for research and analysis.

### Steps:
1. **Setup Preparation:**
   - Ensure `compute_3dmdoi.py` has access to `dermis_pdf.mat` and `dermis_weight.mat`.
   - If `correction_mua.npz` and `correction_mus.npz` are not available, generate these files with the tag "pure" before proceeding with other cases.

2. **Simulated Data Utilization:**
   - Navigate to the `simulated_data` folder, which includes three cases: pure, shallow, and pigment.
     - **Pure:** For correction generation.
     - **Shallow:** Simulates pigment penetration up to 1mm.
     - **Pigment:** Simulates deeper pigment penetration up to 3mm.

3. **Environment Configuration:**
   - Set up the working environment using the `requirements.txt` file.
   - Adjust `compute_3dmdoi.py` (line 585) by setting `save_tag` to "pure", "pigment", or "shallow" based on the test case.

4. **Execution of Codes:**
   - Utilize `compute_3dmdoi.py` to perform 3D-mDOI computation, producing `mua.npz` and `mus.npz`.
   - Utilize `converter3D.py` to convert `mua.npz` files into TIFFs for further processing and visualization through Imaris viewer.
   - Perform quantitative analysis using `quantitative_measurement.py`, reproducing measurements as detailed in the project's main draft.

### Environment Requirements:
- **MATLAB:** Essential for handling `.mat` files.
- **Python:** Prepare as specified in `requirements.txt` for script execution.
- **Simulated Data:** Must be organized according to guidelines for `compute_3dmdoi.py`.

---

## Task 3: Physical Phantom Reconstruction
**Objective:** Conduct physical phantom reconstruction by employing detailed imaging and analysis techniques to create accurate 3D models.

### Steps:
1. **Setup Preparation:**
   - Verify the presence of `pdf_map.mat`, `w_map.mat`, and `correction.mat` within the Matlab CODE folder for seamless operation.
   - Gather the required raw image data from the `OPTIMAP_data` folder, which includes `pa_imgs.mat`, `pb_imgs.mat`, `pc_imgs.mat`, `pd_imgs.mat`, and `bg_imgs.mat`. These files contain cells representing reemitted diffuse data at varying exposures for comprehensive analysis.

2. **Execution of Codes:**
   - Initiate the `main.m` file to begin the reconstruction process, which processes the 3D-mDOI pipeline. Analyze the distribution of light spots for detailed feature extraction. The computed `mua` and `mus` values are stored in `phantom_result.mat` for subsequent evaluation.
   - Post-reconstruction analysis is supported by `analysis.py`, requiring Python 3.7 environment with dependencies.

3. **Result Data Utilization:**
   - Maintain a structured directory with `phantom_result.mat`, `fem_result.npz`, and `bg_Ref.mat` in the `result_data` folder for analysis. The `ims_file` folder should contain IMS files for 3D-mDOI, Finite Element Method (FEM), and Ground-truth visualization.

### Environment Requirements:
- **MATLAB:** Utilize MATLAB R2017a or a more recent version, adjusting functions as necessary to ensure compatibility.
- **Imaris Viewer:** Utilize the Imaris file converter to transfer output files into the IMS format, and employ the Imaris viewer for visual analysis of the 3D models.
  - For detailed guidance, visit the official Imaris learning pages:
    - [Imaris file converter](https://imaris.oxinst.com/learning/view/article/importing-data-into-imaris)
    - [Imaris viewer](https://imaris.oxinst.com/imaris-viewer)

---

## Additional Information
For detailed information on the code and environment setup, refer to the specific README files within each task folder. Ensure proper environment setup and file locations as outlined to facilitate the successful execution of each task.

---

**Shanshan Cai**
