# make sure install imaris
# !pip install imaris-ims-file-reader
# author Shanshan Cai

import numpy as np
import matplotlib.pyplot as plt
# func of RMSE
def rmse(gt, mdoi, fem, label = 'pigment'):
  print("test label: ", label)
  gt = gt/255.0
  mdoi_remap = (mdoi-mdoi.min())/(mdoi.max()-mdoi.min())*1.0
  rms_doi = np.sqrt(((mdoi_remap - gt[:,17:45,17:45]) ** 2).mean())
  print("doi rms: ",rms_doi)
  # the resolutions of z for gt and mdoi are different
  fem_remap = (fem-fem.min())/(fem.max()-fem.min())*1.0
  ans = []
  for z in range(fem_remap.shape[0]):
      ans.append(np.sqrt(((fem_remap[z,:,:] - gt[z*2,16:45,16:45])** 2).mean()))
  rms_fem = np.asarray(ans).mean()
  print("fem rms: ",rms_fem)

  #func of contrast
# CNR =   Cab/ sigma noise
def contrast_seg(gt, mdoi, fem, mdoi_min = 160 , fem_min = 818, label = 'pigment', *p_max):
  print("test label: ", label)
  mdoi_remap = np.zeros(mdoi.shape, dtype='int')
  mdoi_remap[mdoi>=mdoi_min]=1
  if len(p_max)==2:
    mdoi_remap[mdoi<=p_max[0]]=0
  mdoi = (mdoi-mdoi.min())/(mdoi.max()-mdoi.min())*1.0
  mdoi_sorted = np.sort(mdoi.reshape(-1))
  cut_range = int(mdoi_sorted.shape[0]*0.2) 
  print(cut_range)
  mdoi_ob = np.mean(mdoi_sorted[-cut_range:])
  mdoi_bg = np.mean(mdoi_sorted[:cut_range])
  print("mdoi contrast: ", (mdoi_ob-mdoi_bg)/mdoi_bg)
  # the resolutions of z for gt and mdoi are different
  fem_remap = np.zeros(fem.shape, dtype='int')
  fem_remap[fem>=fem_min]=1
  if len(p_max)==2:
    fem_remap[fem<=p_max[1]]=0
  fem = (fem-fem.min())/(fem.max()-fem.min())*1.0
  fem_sorted = np.sort(fem.reshape(-1))
  cut_range = int(fem_sorted.shape[0]*0.2) 
  fem_ob = np.mean(fem_sorted[-cut_range:])
  fem_bg = np.mean(fem_sorted[:cut_range])
  print("fem contrast: ", (fem_ob-fem_bg)/fem_bg)


  # func of bhattacharyya distance
# source: https://www.kaggle.com/debanga/statistical-distances
def bhattacharyya_distance(distribution1: "dict", distribution2: "dict",) -> int:
    """ Estimate Bhattacharyya Distance (between General Distributions)
    
    Args:
        distribution1: a sample distribution 1
        distribution2: a sample distribution 2
    
    Returns:
        Bhattacharyya distance
    """
    sq = 0
    for i in range(len(distribution1)):
        sq  += np.sqrt(distribution1[i]*distribution2[i])
    
    return -np.log(sq)

def b_dist(gt, mdoi, fem, label = 'pigment'):
  print("test label: ", label)
  mdoi_p = np.histogram(mdoi.reshape(-1), bins=10)[0]
  mdoi_p = mdoi_p/np.sum(mdoi_p)
  gt_remap = gt[:,17:45,17:45]
  gt_p = np.histogram(gt_remap.reshape(-1), bins=10)[0]
  gt_p = gt_p/np.sum(gt_p) 
  doi_dist = bhattacharyya_distance(mdoi_p, gt_p)
  print("doi b_dist: ",doi_dist)

  # the resolutions of z for gt and mdoi are different
  gt_remap = gt[::2,16:45,16:45]
  gt_remap = gt_remap[:fem.shape[0],:,:]
  gt_p = np.histogram(gt_remap.reshape(-1), bins=10)[0]
  gt_p = gt_p/np.sum(gt_p) 
  fem_p = np.histogram(fem.reshape(-1), bins=10)[0]
  fem_p = fem_p/np.sum(fem_p)
  fem_dist = bhattacharyya_distance(fem_p, gt_p)
  print("fem b_dist: ",fem_dist)

# b_dist(gt, mdoi, fem)
# # Distribution 3
# d1 = np.random.rand(1,1000)
# p1 = np.histogram(d1,100)[0]
# p1 = p1 / np.sum(p1)

# # Distribution 4
# d2 = np.random.rand(1,1000)
# p2 = np.histogram(d2,100)[0]
# p2 = p2 / np.sum(p2)
# # Our implementation (General)
# distance = bhattacharyya_distance(p1, p2)
# print(f"Ours (General)  : {distance}")


# visualization
def visualization(gt, mdoi, fem, label = 'pigment'):
  print("test label: ", label)
  mdoi_remap = (mdoi-mdoi.min())/(mdoi.max()-mdoi.min())*255.0
  # the resolutions of z for gt and mdoi are different
  fem_remap = (fem-fem.min())/(fem.max()-fem.min())*255.0
  ans = []
  for z in range(fem_remap.shape[0]):
      plt.imshow(fem_remap[z,:,:])
      plt.title('fem')
      plt.show()
      plt.imshow(mdoi_remap[2*z,:,:])
      plt.title('mdoi')
      plt.show()
      plt.imshow(gt[z*2,16:45,16:45])
      plt.title('gt')
      plt.show()
 
# visualization
def visualize_seg(gt, mdoi, fem, mdoi_min = 160 , fem_min = 818, label = 'pigment'):
  print("test label: ", label)
  gt_remap = np.zeros(gt.shape, dtype='int')
  gt_remap[gt==255]=1
  gt_remap=gt_remap[::2,16:45,16:45]
  gt_remap = gt_remap[:fem.shape[0],:,:]
  mdoi_remap = np.zeros(mdoi.shape, dtype='int')
  mdoi_remap[mdoi>=mdoi_min]=1 
  mdoi_remap=mdoi_remap[::2,:,:]
  mdoi_remap = mdoi_remap[:fem.shape[0],:,:]
  fem_remap = np.zeros(fem.shape, dtype='int')
  fem_remap[fem>=fem_min]=1
  for z in range(fem_remap.shape[0]):
      plt.imshow(fem_remap[z,:,:])
      plt.title('fem')
      plt.show()
      plt.imshow(mdoi_remap[z,:,:])
      plt.title('mdoi')
      plt.show()
      plt.imshow(gt_remap[z,:,:])
      plt.title('gt')
      plt.show()


# reference: https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py
# https://www.cns.nyu.edu/pub/eero/wang03b.pdf
from scipy import signal
from scipy import ndimage


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def ssim_c(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
        
def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image 
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity 
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on 
    Signals, Systems and Computers, Nov. 2003 
    
    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim_c(im1, im2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))

#func of SSIM 
from skimage.metrics import structural_similarity as ssim
def ssim3D(gt, mdoi, fem, label = 'pigment'):
  print("test label: ", label)
  mdoi_remap = (mdoi-mdoi.min())/(mdoi.max()-mdoi.min())*255.0
  # mdoi_ssim = ssim(mdoi_remap,gt[:,17:45,17:45])
  ans = []
  for z in range(mdoi_remap.shape[0]):
      ans.append(msssim(mdoi_remap[z,:,:],gt[z,17:45,17:45]))
  mdoi_ssim = np.asarray(ans).mean()

  print("doi ssim: ",mdoi_ssim)
  # the resolutions of z for gt and mdoi are different
  fem_remap = (fem-fem.min())/(fem.max()-fem.min())*255.0
  gt_remap=gt[::2,16:45,16:45]
  gt_remap = gt_remap[:fem.shape[0],:,:]
  ans = []
  for z in range(fem_remap.shape[0]):
      ans.append(msssim(fem_remap[z,:,:],gt_remap[z,:,:]))
  fem_ssim = np.asarray(ans).mean()
  # fem_ssim = ssim(fem_remap,gt_remap)
  print("fem ssim: ",fem_ssim)


  # func of seg-expected depth
def expected_depth(gt, mdoi, fem, mdoi_min = 160 , fem_min = 818, label = 'pigment', *p_max):
  print("test label: ", label)
  th = 0.99
  gt_resolution = 0.2
  gt = gt/255.0
  gt_idx  = np.where(gt==1)
  # first ind is z axis
  gt_z = np.asarray(sorted(gt_idx[0]))
  gt_n = int(gt_z.shape[0]*th) 
  gt_max_z = np.mean(gt_z[gt_n:])*gt_resolution
  print("gt expected depth seg: ",gt_max_z)

  mdoi_resolution = 0.2
  mdoi_remap = np.zeros(mdoi.shape, dtype='int')
  mdoi_remap[mdoi>=mdoi_min]=1
  if len(p_max)==2:
    mdoi_remap[mdoi<=p_max[0]]=0
  mdoi_idx  = np.where(mdoi_remap==1)
  mdoi_z = np.asarray(sorted(mdoi_idx[0]))
  mdoi_n = int(mdoi_z.shape[0]*th) 
  mdoi_max_z = np.mean(mdoi_z[mdoi_n:])*mdoi_resolution  
  print("doi expected depth seg: ",mdoi_max_z)
  # the resolutions of z for gt and mdoi are different
  fem_resolution = 0.5
  fem_remap = np.zeros(fem.shape, dtype='int')
  fem_remap[fem>=fem_min]=1
  if len(p_max)==2:
    fem_remap[fem<=p_max[1]]=0
  fem_idx  = np.where(fem_remap==1)
  fem_z = np.asarray(sorted(fem_idx[0]))
  fem_n = int(fem_z.shape[0]*th) 
  fem_max_z = np.mean(fem_z[fem_n:])*fem_resolution   
  print("fem expected depth seg: ",fem_max_z)

# func of Dice
def dice(gt, mdoi, fem, mdoi_min = 160 , fem_min = 818, label = 'pigment', *p_max):
  print("test label: ", label)
  gt_remap = np.zeros(gt.shape, dtype='int')
  gt_remap[gt==255]=1
  gt_remap=gt_remap[:,17:45,17:45]
  mdoi_remap = np.zeros(mdoi.shape, dtype='int')
  mdoi_remap[mdoi>=mdoi_min]=1
  if len(p_max)==2:
    mdoi_remap[mdoi<=p_max[0]]=0
  dice_mdoi = np.sum(mdoi_remap[gt_remap==1])*2.0 / (np.sum(mdoi_remap) + np.sum(gt_remap))
  print("doi dice: ",dice_mdoi)
  # the resolutions of z for gt and mdoi are different
  gt_remap = np.zeros(gt.shape, dtype='int')
  gt_remap[gt==255]=1
  gt_remap=gt_remap[::2,16:45,16:45]
  gt_remap = gt_remap[:fem.shape[0],:,:]
  fem_remap = np.zeros(fem.shape, dtype='int')
  fem_remap[fem>=fem_min]=1
  if len(p_max)==2:
    fem_remap[fem<=p_max[1]]=0
  dice_fem = np.sum(fem_remap[gt_remap==1])*2.0 / (np.sum(fem_remap) + np.sum(gt_remap))
  print("fem dice: ",dice_fem)


# func volume ratio
def vr(gt, mdoi, fem, mdoi_min = 160 , fem_min = 818, label = 'pigment', *p_max):
  print("test label: ", label)
  gt_remap = np.zeros(gt.shape, dtype='int')
  gt_remap[gt==255]=1
  gt_remap=gt_remap[:,17:45,17:45]
  mdoi_remap = np.zeros(mdoi.shape, dtype='int')
  mdoi_remap[mdoi>=mdoi_min]=1
  if len(p_max)==2:
    mdoi_remap[mdoi<=p_max[0]]=0
  mdoi_tp =  np.logical_and(mdoi_remap ==1 , gt_remap ==1)
  mdoi_tn =  np.logical_and(mdoi_remap ==0 , gt_remap ==0)
  sensitivity_mdoi = np.count_nonzero(mdoi_tp)/np.count_nonzero(gt_remap)
  specificity_mdoi = np.count_nonzero(mdoi_tn)/np.count_nonzero(1-gt_remap)
  print("doi sensitivity seg: ",sensitivity_mdoi)
  print("doi specificity seg: ",specificity_mdoi)
  print("doi volume ratio: ", np.count_nonzero(mdoi_tp[0,:,:])/max(np.count_nonzero(gt_remap[0,:,:]),1))


  # # the resolutions of z for gnp.sumt and mdoi are different
  gt_remap = np.zeros(gt.shape, dtype='int')
  gt_remap[gt==255]=1
  gt_remap=gt_remap[::2,16:45,16:45]
  gt_remap = gt_remap[:fem.shape[0],:,:]
  fem_remap = np.zeros(fem.shape, dtype='int')
  fem_remap[fem>=fem_min]=1
  if len(p_max)==2:
    fem_remap[fem<=p_max[1]]=0
  fem_tp =  np.logical_and(fem_remap ==1 , gt_remap ==1)
  fem_tn =  np.logical_and(fem_remap ==0 , gt_remap ==0)
  sensitivity_fem = np.count_nonzero(fem_tp)/np.count_nonzero(gt_remap)
  specificity_fem = np.count_nonzero(fem_tn)/np.count_nonzero(1-gt_remap)
  print("fem sensitivity seg: ",sensitivity_fem)
  print("fem specificity seg: ",specificity_fem)
  print("fem volume ratio: ", np.count_nonzero(fem_tp[0,:,:])/max(np.count_nonzero(gt_remap[0,:,:]),1))






from imaris_ims_file_reader.ims import ims
import os

dir = os.getcwd() + "/result_data/"
# test case of pigement 
data = ims(dir+'pigment_gt.ims')
gt = np.squeeze(np.asarray(data[0,0,:,:,:]))
data = ims(dir+'nirfast_pigment_mua.ims')
fem = np.squeeze(np.asarray(data[0,0,:,:,:]))
data = ims(dir+'pigment_mua.ims')
mdoi = np.squeeze(np.asarray(data[0,0,:,:,:]))
rmse(gt, mdoi, fem)
b_dist(gt, mdoi, fem)
mdoi_th = 150
fem_th = 1909
dice(gt, mdoi, fem,mdoi_th,fem_th)
contrast_seg(gt, mdoi, fem,mdoi_th,fem_th)
vr(gt, mdoi, fem,mdoi_th,fem_th)
ssim3D(gt, mdoi, fem)
expected_depth(gt, mdoi, fem,mdoi_th,fem_th)


# test case of shallow
data = ims(dir+'shallow_gt.ims')
gt = np.squeeze(np.asarray(data[0,0,:,:,:]))
data = ims(dir+'nirfast_shallow_mua.ims')
fem = np.squeeze(np.asarray(data[0,0,:,:,:]))
data = ims(dir+'shallow_mua.ims')
mdoi = np.squeeze(np.asarray(data[0,0,:,:,:]))
mdoi_th = 300
fem_th = 2000
tag = 'shallow'
rmse(gt, mdoi, fem,tag)
b_dist(gt, mdoi, fem,tag)
dice(gt, mdoi, fem,mdoi_th,fem_th,tag)
contrast_seg(gt, mdoi, fem,mdoi_th,fem_th,tag)
vr(gt, mdoi, fem,mdoi_th,fem_th,tag)
ssim3D(gt, mdoi, fem,tag)
expected_depth(gt, mdoi, fem,mdoi_th,fem_th,tag)





