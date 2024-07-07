import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
from matplotlib.pylab import scatter
from pylab import plot, show, savefig, xlim, figure,  ylim, legend, boxplot, setp, axes

import matplotlib.pyplot as plt
import os


folder_name = os.getcwd() + "/result_data/"
file_name = folder_name + "phantom_result.mat"
mua = sio.loadmat(file_name)['multi_mua']
mus = sio.loadmat(file_name)['multi_mus']
file_name = folder_name + 'bg_Ref.mat'
ref = sio.loadmat(file_name)['ref']
file_name = folder_name + 'fem_result.npz'
fem = np.load(file_name)['result']


# print out the reference of the analysis roi
off = 0

rsize = 45
roi_range=[[78+off,78+rsize-off,106+off,106+rsize-off],
    [80+off,80+rsize-off,585+off,585+rsize-off],
    [392+off,392+rsize-off,92+off,92+rsize-off],
    [715+off,715+rsize-off,85+off,85+rsize-off],
    [717+off,717+rsize-off,564+off,564+rsize-off],
    [320+off*3,320+rsize-off*3,400+off*2,400+rsize-off*3],
    [20+off,20+rsize-off,106+off,106+rsize-off],
    [80+off,80+rsize-off,500+off,500+rsize-off],
    [320+off,320+rsize-off,92+off,92+rsize-off],
    [650+off,650+rsize-off,85+off,85+rsize-off],
    [720+off,720+rsize-off,650+off,650+rsize-off],
    [320+off,320+rsize-off,640+off,640+rsize-off]]


plt.imshow(ref)
# Get the current reference
ax = plt.gca()
for i in range(6):
  o = roi_range[i]
  rect = Rectangle((o[0],o[2]),o[1]-o[0],o[3]-o[2],linewidth=1,edgecolor='r',facecolor='none')
  ax.add_patch(rect)
  u = roi_range[i+6]
  rect = Rectangle((u[0],u[2]),u[1]-u[0],u[3]-u[2],linewidth=1,edgecolor='yellow',facecolor='none')
  ax.add_patch(rect)
plt.show()



mua_roi,mus_roi = [],[]
fem_mua,fem_mus =  [],[]
for a in roi_range:
  # plt.imshow(ref[a[2]:a[3],a[0]:a[1]])
  # plt.show()
  mua_roi.append(mua[a[2]:a[3],a[0]:a[1],:])
  mus_roi.append(mus[a[2]:a[3],a[0]:a[1],:])
  fem_mua.append(fem[0,a[2]//8:a[3]//8,a[0]//8:a[1]//8,:])
  fem_mus.append(fem[1,a[2]//8:a[3]//8,a[0]//8:a[1]//8,:])




# print out the quantitative analysis
# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')

    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['fliers'][1], color='red')
    setp(bp['medians'][1], color='red')


num_feat = 6

def compute_avg(mua_roi, mus_roi):
  z_binsize = 0.05

  depth = np.arange(0,mua_roi[0].shape[2])
  depth = depth *z_binsize
  w = np.exp(-0.1*depth)

  avg_dmus = []
  avg_dmua = []


  for i in range(len(mus_roi)):
    avg_dmus.append(np.mean(np.abs(mus_roi[i])*w,axis=(2)))
    avg_dmua.append(np.mean(np.abs(mua_roi[i])*w,axis=(2)))

  # print()

  # avg_dmus = np.asarray(avg_dmus)
  # avg_dmua = np.asarray(avg_dmua)

  for i in range(num_feat):
    avg_dmus[i] = avg_dmus[i] / np.mean(avg_dmus[i+num_feat] ,axis = (0,1))
    avg_dmua[i] = avg_dmua[i] / np.mean(avg_dmua[i+num_feat] ,axis = (0,1))
  return avg_dmua[:num_feat], avg_dmus[:num_feat]

mdoi_mua, mdoi_mus  = compute_avg(mua_roi, mus_roi)
afem_mua, afem_mus  = compute_avg(fem_mua, fem_mus)


expected_mua = (np.asarray([0.19,0.36, 0.88,0.36,0.36,0.33])+0.5)/0.5
expected_mus = np.asarray([90,90,97,97,93,53])/90

fig = figure()
ax = axes()




# first boxplot pair
for i in range(num_feat):
  bp = boxplot([mdoi_mua[i].reshape(-1),afem_mua[i].reshape(-1)], positions = [i*3+1, i*3+2], widths = 0.6)
  setBoxColors(bp)


# set axes limits and labels
xlim(0,18)
ax.set_xticks([1.5+i*3 for i in range(num_feat)])
ax.set_xticklabels(['{}'.format(i+1) for i in range(num_feat)])


# draw temporary red and blue lines and use them to create a legend
hB, = plot([1,1],'b-')
hR, = plot([1,1],'r-')
hG, = plot([1,1],'g--')
legend((hB, hR,hG),('3D-mDOI', 'FEM', 'Expected'),fontsize=13)
hB.set_visible(False)
hR.set_visible(False)
hG.set_visible(False)

ax.set_ylabel('Relative ua ratio [$cm^{-1}$/$cm^{-1}$]',fontsize=13)
ax.set_xlabel('Feature index [AU]',fontsize=13)
# ax.set_title('Comparison relative ua ratio between 3D-mDOI and FEM')
ax.set_title('Comparison of relative ua ratio',fontsize=15)

plt.xticks(fontsize=13) # Increase font size for x-axis ticks
plt.yticks(fontsize=13)


for i in range(num_feat):
  plt.hlines(expected_mua[i], xmin=i*3+0.5, xmax=i*3+2.5, color='green', linestyles='--')

plt.show()



fig = figure()
ax = axes()



# first boxplot pair
for i in range(num_feat):
  bp = boxplot([mdoi_mus[i].reshape(-1),afem_mus[i].reshape(-1)], positions = [i*3+1, i*3+2], widths = 0.6)
  setBoxColors(bp)


# set axes limits and labels
xlim(0,18)
ax.set_xticks([1.5+i*3 for i in range(num_feat)])
ax.set_xticklabels(['{}'.format(i+1) for i in range(num_feat)])


ax.set_ylim([0.5,1.3])


# draw temporary red and blue lines and use them to create a legend
hB, = plot([1,1],'b-')
hR, = plot([1,1],'r-')
hG, = plot([1,1],'g--')
legend((hB, hR,hG),('3D-mDOI', 'FEM', 'Expected'),fontsize=13)
hB.set_visible(False)
hR.set_visible(False)
hG.set_visible(False)

ax.set_ylabel('Relative us ratio [$cm^{-1}$/$cm^{-1}$]',fontsize=13)


ax.set_xlabel('Feature index [AU]',fontsize=13)
# ax.set_title('Comparison relative ua ratio between 3D-mDOI and FEM')
ax.set_title('Comparison of relative us ratio',fontsize=15)

plt.xticks(fontsize=13) # Increase font size for x-axis ticks
plt.yticks(fontsize=13)



for i in range(num_feat):
  plt.hlines(expected_mus[i], xmin=i*3+0.5, xmax=i*3+2.5, color='green', linestyles='--')




z_binsize = 0.05
color = ['b','g','r','c','m','y']
depth = np.arange(0,mua_roi[0].shape[2])
depth = depth *z_binsize

fig, axs = plt.subplots(1, 6, figsize=(24,4))
fig.suptitle('Relative ua ratio as a function of depth', fontsize=20)

for i in range(num_feat):
    ax = axs[i]
    ratio =  np.mean(mua_roi[i+num_feat],axis = (0,1))
    d = np.mean(mua_roi[i]/ratio,axis = (0,1))
    s = np.std(mua_roi[i]/ratio,axis = (0,1))
    ax.set_ylim(0.5,3)
    ax.errorbar(depth,d, yerr=s, label='Feat.{}'.format(i+1),c = color[i])
    ax.legend(loc='upper right', fontsize=15)
    # ax.set_xlabel('Depth [cm]', fontsize=15)
    if not i:
      ax.set_ylabel('Relative ua ratio [$cm^{-1}$/$cm^{-1}$]', fontsize=15)

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

axs[-1].set_xlabel('Depth [cm]', fontsize=15)
plt.show()


fig, axs = plt.subplots(1, 6, figsize=(24,4), sharex=True)

fig.suptitle('Relative us ratio as a function of depth', fontsize=20)

for i in range(num_feat):
    ax = axs[i]
    ratio =  np.mean(mus_roi[i+num_feat],axis = (0,1))
    d = np.mean(mus_roi[i]/ratio,axis = (0,1))
    s = np.std(mus_roi[i]/ratio,axis = (0,1))
    ax.set_ylim([0.5,1.4])
    ax.errorbar(depth,d, yerr=s, label='Feat.{}'.format(i+1),c = color[i])
    ax.legend(loc='upper right', fontsize=15)
    # ax.set_xlabel('Depth [cm]', fontsize=15)
    if not i:
      ax.set_ylabel('Relative us ratio [$cm^{-1}$/$cm^{-1}$]', fontsize=15)

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

# for ax in axs[:-1]:
#     ax.set_xlabel('')  # This hides the x-axis labels
#     ax.tick_params(labelbottom=False)


axs[-1].set_xlabel('Depth [cm]', fontsize=15)

plt.show()


