# python version of the cosmetics code
# Shanshan Cai
# mm
# 2019-06-1
# current perfect version

import time as ti
import scipy.io as sio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
from sklearn import mixture
from scipy.ndimage import zoom
from scipy.optimize import curve_fit

from pyspark import SparkContext
from itertools import product
from sklearn.decomposition import PCA
import joblib
import os
from mpl_toolkits.mplot3d import Axes3D

class roi_range:
    def __init__(self, u=550, d=750, l=900, r=1200):
        self.u = u
        self.d = d
        self.l = l
        self.r = r


old_dataRange = roi_range(u=550, d=750, l=900, r=1200)
class sys_params:
    def __init__(self, exp_times, pixel_size, banana_size=0.5, neighborhood_size=16, fitting_window=5):
        self.exp_times = exp_times
        self.pixel_size = pixel_size
        self.banana_size = banana_size
        self.neighborhood_size = neighborhood_size
        self.patch_size = neighborhood_size * 2 + 1
        self.fitting_window = fitting_window
        self.p = self.cal_dist(self.neighborhood_size, self.pixel_size)
        self.am = 1.3316
        self.offset = 4.4135
        self.hMask = np.concatenate((np.ones((self.fitting_window, self.patch_size)),
                                     np.zeros(
                                         (self.patch_size - self.fitting_window * 2, self.patch_size)),
                                     np.ones((self.fitting_window, self.patch_size))), axis=0)
        self.vMask = np.transpose(self.hMask)
        self.hMaskIndex = self.hMask == 1
        self.vMaskIndex = self.vMask == 1
        self.zResultion = 0.2  # mm
        self.epsilon = 1e-6

    def cal_dist(self, dSize, ratio):
        var = np.arange(-dSize, dSize + 1)
        X, Y = np.meshgrid(var, var)
        dist = np.sqrt(X ** 2 + Y ** 2) * ratio
        return dist

    def set_Am(self, Am, Offset):
        self.am = Am
        self.offset = Offset

    def load_Bananas(self):
        pdf_map = sio.loadmat('dermis_pdf')
        self.pdf_map = pdf_map['pdf_map']
        w_map = sio.loadmat('dermis_weight')
        self.w_map = np.squeeze(w_map['w_map'])
        self.zslices = self.pdf_map[-1, -1].shape[2]
        self.xslices = self.pdf_map.shape[0] - 1

    def cart2pol(self, x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    # change the banana to fit the setup now
    def convert_Bananas(self):
        self.load_Bananas()
        # todo zslice will change -> change the parameters in class
        offset = (self.xslices - self.neighborhood_size)
        new_weight = np.zeros((self.neighborhood_size + 1, self.neighborhood_size + 1))
        new_banana = dict()
        for y_idx  in range(self.neighborhood_size + 1):
            for x_idx in range(self.neighborhood_size + 1):
                new_banana[y_idx,x_idx] = self.pdf_map[y_idx+offset,x_idx+offset][offset:-offset,offset:-offset,:]
                new_weight[y_idx,x_idx] = self.w_map[y_idx+offset,x_idx+offset]
        self.w_map = new_weight
        self.pdf_map = new_banana
        self.generate_w3d()

    def generate_w3d(self):
        self.w3D = np.zeros((self.patch_size, self.patch_size, self.zslices))
        for y in range(self.patch_size):
            for x in range(self.patch_size):
                y_new = self.neighborhood_size - y
                x_new = self.neighborhood_size - x
                if y_new == 0 and x_new == 0:
                    pass
                pdf = self.pdf_map[(np.abs(y_new), np.abs(x_new))]
                w = self.w_map[(np.abs(y_new), np.abs(x_new))]
                if y_new < 0:
                    # flip
                    pdf = pdf[::-1, :, :]
                if x_new < 0:
                    pdf = pdf[:, ::-1, :]
                self.w3D += pdf * w






class fit_params:
    def __init__(self, mus=1.0, mua=0.0):
        self.mus = mus
        self.mua = mua


# unit mm
red = fit_params(mua=0.001, mus=19.53)
green = fit_params(mua=0.04, mus=35.65)
blue = fit_params(mua=0.008, mus=57.97)
fitting_bounds = ([0, 0], [15, 40])
light = red
# sys setup
exp_times = [50, 150, 300]
pixel_size = 0.5


def avg_tiffs(exp_times, save_loc='', name='phantom', color='red', patterns=np.arange(1, 36, 1), num_imgs=10):
    # load tiffs
    data = dict()
    file_names = save_loc + "{}_{}_p{}_{}ms_{:03d}.tif"
    avg_im = np.array([])
    for p in patterns:
        var_exp = []
        for t in exp_times:
            for i in range(num_imgs):
                fname = file_names.format(name, color, t, p, i)
                im = cv2.imread(fname, -1)
                if avg_im.size > 0:
                    avg_im += im / num_imgs
                else:
                    avg_im = im / num_imgs
            var_exp.append(avg_im)
        data[p] = var_exp
    return data


def HDR_generation(imgs, bg_imgs, exp_times, base_index=0, MaximumLimit=np.power(2, 16), min_thre=5, bVisual=False):
    # init
    img_size = imgs[0].shape
    someUnderExposed = np.zeros(img_size, dtype=bool)
    someOverExposed = np.zeros(img_size, dtype=bool)
    someProperlyExposed = np.zeros(img_size, dtype=bool)
    properlyExposedCount = np.zeros(img_size, dtype='uint8')
    hdr = np.zeros(img_size, dtype='uint16')
    relExposure = np.asarray(exp_times) / exp_times[base_index]
    # calculate the map
    for ldr, MinimumLimit, t in zip(imgs, bg_imgs, relExposure):
        underExposed = ldr < MinimumLimit + min_thre
        someUnderExposed = underExposed | someUnderExposed
        overExposed = ldr > MaximumLimit - min_thre
        someOverExposed = someOverExposed | overExposed
        properlyExposed = ~(underExposed | overExposed)
        someProperlyExposed = someProperlyExposed | properlyExposed
        properlyExposedCount[properlyExposed] = properlyExposedCount[properlyExposed] + 1
        ldr[~properlyExposed] = 0
        hdr = hdr + ldr / t

    hdr = hdr / np.clip(properlyExposedCount, 1, None)

    hdr[someOverExposed & ~someUnderExposed & ~
    someProperlyExposed] = np.max(hdr[someProperlyExposed])
    hdr[someUnderExposed & ~someOverExposed & ~someProperlyExposed] = 0

    hdr = hdr.astype('uint16')

    fillMask = someUnderExposed & someOverExposed & ~someProperlyExposed
    if np.any(fillMask):
        fillMask = fillMask.astype('uint8') * 255
        hdr = cv2.inpaint(hdr, fillMask, 3, cv2.INPAINT_TELEA)

    hdr = np.clip(hdr.astype('int16') -
                  bg_imgs[base_index].astype('int16'), 0, None)
    hdr = hdr.astype('uint16')
    if bVisual == True:
        show_uint16(hdr)
    return hdr


def local_maximum(img, threshold=300, bVisual=False):
    img_size = img.shape
    data_max = filters.maximum_filter(img, sys_args.neighborhood_size)
    maxima = (img == data_max)
    data_min = filters.minimum_filter(img, sys_args.neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) // 2
        y_center = (dy.start + dy.stop - 1) // 2
        # check the feasibility of the locs
        if x_center + sys_args.neighborhood_size < img_size[
            1] and x_center - sys_args.neighborhood_size >= 0 and y_center + sys_args.neighborhood_size < img_size[
            0] and y_center - sys_args.neighborhood_size >= 0:
            x.append(x_center)
            y.append(y_center)

    if bVisual == True:
        plt.imshow(img, cmap="gray", vmin=0, vmax=4096)
        plt.autoscale(False)
        plt.plot(x, y, 'ro')
        plt.show()
    return list(zip(y, x))


def collect_patches(img, locs, bVisual=False):
    patchs = []
    y,x=locs[0]

    patchs.append(img[y - sys_args.neighborhood_size:y + sys_args.neighborhood_size + 1,
                  x - sys_args.neighborhood_size: x + sys_args.neighborhood_size + 1])
    if bVisual == True:
        plot_surf(patchs[0])
    return patchs


# def load_
def show_uint16(img):
    plt.imshow(img, cmap="gray", vmin=0, vmax=4096)
    plt.show()


def plot_surf(img):
    img_size = img.shape
    X, Y = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, img, cmap='gray')
    plt.show()


def select_roi(img, roi):
    return img[roi.u:roi.d, roi.l:roi.r]


def diffusion_theory(p, mua, mus):
    g = 0.9
    nel = 1.33
    rd = -1.44 * np.power(nel, -2) + 0.71 * \
         np.power(nel, -1) + 0.668 + 0.0636 * nel
    A = (1 + rd) / (1 - rd)
    D = 1 / (3 * (mua + (1 - g) * mus))
    zb = 2 * A * D
    mus_reduce = mus * (1 - g)
    mu_eff = np.sqrt(3 * mua * (mua + mus_reduce))
    mut = mus_reduce + mua
    z0 = 1 / mut
    r1 = np.sqrt(z0 ** 2 + p ** 2)
    r2 = np.sqrt((z0 + 2 * zb) ** 2 + p ** 2)
    albedo = mus_reduce / (mus_reduce + mua)
    return sys_args.am * np.log10(
        albedo / (4 * np.pi) * (z0 * (mu_eff + (1 / r1)) * np.exp(-mu_eff * r1) / (r1 ** 2) + (z0 + 2 * zb) *
                                (mu_eff + 1 / r2) * np.exp(-mu_eff * r2) / (r2 ** 2))) + sys_args.offset


def fitting_RTE(p, r):
    popt, pcov = curve_fit(diffusion_theory, p, r, p0=[light.mua, light.mus], bounds=fitting_bounds)
    perr = np.mean(np.sqrt(np.diag(pcov)))
    return popt, perr


def fit_patch(R, Am, offset):
    return Am * R + offset


def fitting_Am(Fxy, R):
    F_max = np.max(Fxy) - np.max(R)
    F_min = np.min(Fxy) - np.min(R)
    init = (F_max - F_min, F_min)
    popt, pcov = curve_fit(fit_patch, Fxy, R, p0=init)
    perr = np.mean(np.sqrt(np.diag(pcov)))
    return popt, perr


def get_Am_map(R, patch):
    popt, perr = fitting_Am(R, patch)
    return popt, perr


def get_Am_reduce(x, y):
    popt1, perr1 = x[0], x[1]
    popt2, perr2 = y[0], y[1]

    popt1islist = isinstance(popt1, list)
    popt2islist = isinstance(popt2, list)

    perr1islist = isinstance(perr1, list)
    perr2islist = isinstance(perr2, list)

    popts = []
    perrs = []
    if popt1islist:
        popts += popt1
    else:
        popts += [popt1]

    if popt2islist:
        popts += popt2
    else:
        popts += [popt2]

    if perr1islist:
        perrs += perr1
    else:
        perrs += [perr1]
    if perr2islist:
        perrs += perr2
    else:
        perrs += [perr2]

    return popts, perrs


def get_Am(patches):
    R = diffusion_theory(sys_args.p, red.mua, red.mus)

    cnts = sc.range(0, len(patches))
    cnts_map = cnts.map(lambda x:
                        get_Am_map(R.reshape(-1), patches[x].reshape(-1)))
    result = cnts_map.reduce(lambda x, y: get_Am_reduce(x, y))
    popts = result[0]
    perrs = result[1]

    #  gaussian model to do the prefitting
    if len(popts.shape)==1:
        sys_args.set_Am(popts[0], popts[1])
    else:
        g = mixture.GaussianMixture()
        g.fit(popts)
        param = np.squeeze(g.means_)
        print(param[0], param[1])
        sys_args.set_Am(param[0], param[1])


def get_Fitting2Typers_Vmap_v2(patches, sys_args, patch_n, x, y):
    patch = patches[patch_n]
    r = np.squeeze(patch[y - sys_args.fitting_window:y + sys_args.fitting_window + 1, x])
    p_val = np.squeeze(sys_args.p[y - sys_args.fitting_window:y + sys_args.fitting_window + 1, x])
    param, error = fitting_RTE(p_val, r)

    return param, error, patch_n, x, y


def get_Fitting2Typers_Hmap_v2(patches, sys_args, patch_n, x, y):
    patch = patches[patch_n]
    r = np.squeeze(patch[y, x - sys_args.fitting_window:x + sys_args.fitting_window + 1])
    p_val = np.squeeze(sys_args.p[y, x - sys_args.fitting_window:x + sys_args.fitting_window + 1])
    param, error = fitting_RTE(p_val, r)

    return param, error, patch_n, x, y



def get_Fitting2Types_v2(patches, ftype):
    patch_shape = patches[0].shape
    error = np.zeros((len(patches),)+ patch_shape)
    params = np.zeros((len(patches),) + patch_shape + (2,))
    # params = np.repeat(params[:, :, np.newaxis], 2, axis=2)
    if ftype == "Vertical":
        ylist = range(sys_args.fitting_window, patch_shape[0] - sys_args.fitting_window + 1)
        xlist = range(patch_shape[1])
        print("starting vertical parallel")
        xy_product = sc.parallelize(product(range(len(patches)), xlist, ylist))
        xy_product_map = xy_product.map(lambda x: get_Fitting2Typers_Vmap_v2(patches, sys_args, x[0], x[1], x[2]))
        result = xy_product_map.collect()

    elif ftype == "Horizontal":
        xlist = range(sys_args.fitting_window, patch_shape[1] - sys_args.fitting_window + 1)
        ylist = range(patch_shape[0])
        print("starting horizontal parallel")
        xy_product = sc.parallelize(product(range(len(patches)), xlist, ylist))
        xy_product_map = xy_product.map(lambda x: get_Fitting2Typers_Hmap_v2(patches, sys_args, x[0], x[1], x[2]))
        result = xy_product_map.collect()
    else:
        raise ValueError("The type for fitting does not exist")

    for r in result:
        y = r[4]
        x = r[3]
        params[r[2],y, x, :], error[r[2],y, x] = r[0], r[1]

    return params, error


def generate_3dPatch(params):
    mua3D = np.zeros(
        (sys_args.patch_size, sys_args.patch_size, sys_args.zslices))
    mus3D = np.zeros(
        (sys_args.patch_size, sys_args.patch_size, sys_args.zslices))
    # TODO change w to 1/w
    for y in range(params.shape[0]):
        for x in range(params.shape[1]):
            y_new = sys_args.neighborhood_size - y
            x_new = sys_args.neighborhood_size - x
            pdf = sys_args.pdf_map[(np.abs(y_new), np.abs(x_new))]
            w = 1-sys_args.w_map[(np.abs(y_new), np.abs(x_new))]
            if len(pdf.shape) == 3:
                # pdf = pdf[4:37, 4:37, :]
                if y_new < 0:
                    # flip
                    pdf = pdf[::-1, :, :]
                if x_new < 0:
                    pdf = pdf[:, ::-1, :]
                mua3D += pdf * w * params[y, x, 0]
                mus3D += pdf * w * params[y, x, 1]
    mua3D = mua3D * sys_args.w3D
    mus3D = mus3D * sys_args.w3D
    return mua3D, mus3D


def generate_CorrectionMatrix(paramslist, locs,bPCA):
    mua3D = np.zeros((sys_args.patch_size,sys_args.patch_size, sys_args.zslices))
    mus3D = np.zeros((sys_args.patch_size, sys_args.patch_size, sys_args.zslices))
    mua3D_pca=[]
    mus3D_pca=[]
    n=len(locs)
    for i, l in enumerate(locs):
        params3D = generate_3dPatch(paramslist[i])
        mua3D+= params3D[0]/n
        mus3D+= params3D[1]/n
        mua3D_pca.append(params3D[0].reshape(-1))
        mus3D_pca.append(params3D[1].reshape(-1))
    if bPCA:
        pca_mua = PCA(0.8)
        components=pca_mua.fit_transform(mua3D_pca)
        mua3D_denoise = pca_mua.inverse_transform(components)
        pca_mus = PCA(0.2)
        components=pca_mus.fit_transform(mus3D_pca)
        mus3D_denoise = pca_mus.inverse_transform(components)
        mua3D_denoise = np.mean(mua3D_denoise, axis=0)
        mus3D_denoise=np.mean(mus3D_denoise,axis=0)

        np.savez("correction_mua_pca.npz",mua3D_denoise.reshape(mua3D.shape))
        np.savez("correction_mus_pca.npz",mus3D_denoise.reshape(mus3D.shape))
        joblib.dump(pca_mua, 'pca_mua.sav')
        joblib.dump(pca_mus, 'pca_mus.sav')
    else:
        np.savez("correction_mua.npz",mua3D)
        np.savez("correction_mus.npz",mus3D)

    print("save correction matrix")

def generate_PatchesCombination_single(paramslist, locs, roi_size):
    mua3D = np.zeros((roi_size[0],roi_size[1], sys_args.zslices))
    mus3D = np.zeros((roi_size[0], roi_size[1], sys_args.zslices))
    w3D = np.zeros((roi_size[0],roi_size[1], sys_args.zslices))


    for i,l in enumerate(locs):
        params3D = generate_3dPatch(paramslist[i])
        mua,mus=params3D[0],params3D[1]


        mua3D[l[0] - sys_args.neighborhood_size:l[0] + sys_args.neighborhood_size + 1,
                  l[1] - sys_args.neighborhood_size: l[1] + sys_args.neighborhood_size + 1, :] += mua*sys_args.w3D
        mus3D[l[0] - sys_args.neighborhood_size:l[0] + sys_args.neighborhood_size + 1,
                  l[1] - sys_args.neighborhood_size: l[1] + sys_args.neighborhood_size + 1, :] += mus *sys_args.w3D
        w3D[l[0] - sys_args.neighborhood_size:l[0] + sys_args.neighborhood_size + 1,
            l[1] - sys_args.neighborhood_size: l[1] + sys_args.neighborhood_size + 1, :] += sys_args.w3D
    w3D[w3D == 0] = 1
    mua3D = mua3D / w3D
    mus3D = mus3D / w3D
    return mua3D, mus3D


def generate_PatchesCombination_v2(paramslist, locs, roi_size,bPCA):

    mua3D = np.zeros((roi_size[0],roi_size[1], sys_args.zslices))
    mus3D = np.zeros((roi_size[0], roi_size[1], sys_args.zslices))
    w3D = np.zeros((roi_size[0],roi_size[1], sys_args.zslices))
    # here use the fixed correct matrix
    if bPCA:
        correct_mua3D=np.load("correction_mua_pca.npz")['arr_0']
        correct_mus3D = np.load("correction_mus_pca.npz")['arr_0']
        pca_mua = joblib.load("pca_mua.sav")
        pca_mus = joblib.load("pca_mus.sav")
    else:
        correct_mua3D = np.load("correction_mua.npz")['arr_0']
        correct_mus3D = np.load("correction_mus.npz")['arr_0']
    # load the model from disk

    uniform = np.ones(correct_mua3D.shape)

    correct_mua3D[correct_mua3D==0] = 1
    correct_mus3D[correct_mus3D==0] = 1

    weight_mua3D = uniform / correct_mua3D
    weight_mus3D = uniform / correct_mus3D

    for i,l in enumerate(locs):
        params3D = generate_3dPatch(paramslist[i])
        mua,mus=params3D[0],params3D[1]
        if bPCA:
            components = pca_mua.transform(params3D[0].reshape(1,-1))
            components= pca_mua.inverse_transform(components)
            mua=components.reshape(correct_mua3D.shape)

            components = pca_mus.transform(params3D[1].reshape(1,-1))
            components= pca_mus.inverse_transform(components)
            mus=components.reshape(correct_mus3D.shape)

        mua3D[l[0] - sys_args.neighborhood_size:l[0] + sys_args.neighborhood_size + 1,
                  l[1] - sys_args.neighborhood_size: l[1] + sys_args.neighborhood_size + 1, :] += mua*weight_mua3D*sys_args.w3D
        mus3D[l[0] - sys_args.neighborhood_size:l[0] + sys_args.neighborhood_size + 1,
                  l[1] - sys_args.neighborhood_size: l[1] + sys_args.neighborhood_size + 1, :] += mus *weight_mus3D*sys_args.w3D
        w3D[l[0] - sys_args.neighborhood_size:l[0] + sys_args.neighborhood_size + 1,
            l[1] - sys_args.neighborhood_size: l[1] + sys_args.neighborhood_size + 1, :] += sys_args.w3D
    w3D[w3D == 0] = 1
    mua3D = mua3D / w3D
    mus3D = mus3D / w3D
    return mua3D, mus3D

def norm(combine3D):
    y,x,z = combine3D.shape
    for i in range(z):
        layer=combine3D[:,:,i]
        norm_value=np.median(layer)
        combine3D[:, :, i] = layer/norm_value


def visulize_3Dslices(combine3D, btype="mua", slice_num=9):
    # add color bar
    resolution = int(sys_args.zslices / slice_num)

    # crop the center range
    combine3D=combine3D[sys_args.neighborhood_size:-sys_args.neighborhood_size,sys_args.neighborhood_size:-sys_args.neighborhood_size,:]
    # combine3D = combine3D[32:88,32:88,:]

    vmin = np.min(combine3D)
    vmax = np.max(combine3D)
    fig, axes = plt.subplots(nrows=3, ncols=slice_num//3)
    for i,ax in enumerate(axes.flat):
        name = str(round(sys_args.zResultion * i * resolution, 2)) + "mm"
        s=ax.imshow(combine3D[:, :, i * resolution+2],vmin=vmin,vmax=vmax)
        ax.set_title(name)

    fig.suptitle(btype + "results")
    fig.subplots_adjust(right=0.7,hspace=0.01,wspace=0.01,top=0.8)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(s,cax=cbar_ax)
    plt.show()


def get_LightParams(patches):
    # todo if fitting error huge, expand the fitting range and do it again
    # process vertically
    vParams, vError = get_Fitting2Types_v2(patches, "Vertical")
    # process horizontally
    hParams, hError = get_Fitting2Types_v2(patches, "Horizontal")
    tError = vError + hError
    tError[tError == 0] = 1
    # param(n_patch, patch_y, patch_x, 2)
    params = (vParams * hError[:, :, :, np.newaxis] + hParams *
              vError[:, :,:, np.newaxis]) / tError[:, :, :,np.newaxis]
    tError = vError + hError
    tError[:,sys_args.hMaskIndex] *= 2
    tError[:,sys_args.vMaskIndex] *= 2
    params[:,sys_args.hMaskIndex, :] += hParams[:,sys_args.hMaskIndex, :]
    params[:,sys_args.vMaskIndex, :] += vParams[:,sys_args.vMaskIndex, :]
    return params, tError


if __name__ == '__main__':
    save_tag="pigment"
    bTest=True

    if save_tag == "pure":
        bCorrect = True
    else:
        bCorrect = False
    sys_args = sys_params(exp_times, pixel_size)
    sys_args.convert_Bananas()
    if bTest:
        t0 = ti.perf_counter()
        sc = SparkContext("local[*]", "cosmetics")
        locs = []
        patches_all = []
        npattern = 64

        dir = os.getcwd() + "/simulated_data/"

        pattern_file = dir + "/{}/{}_".format(save_tag,save_tag)
        t1 = ti.perf_counter()
        for p in range(1, 1 + npattern):
            file_name = pattern_file + str(p) + ".mat"
            imgs = sio.loadmat(file_name)['U']
            loc_tuple = np.unravel_index(np.argmax(imgs), imgs.shape)
            loc = [[loc_tuple[0],loc_tuple[1]]]

            if loc[0][0] < sys_args.neighborhood_size:
                loc[0][0] = sys_args.neighborhood_size
            if loc[0][1] < sys_args.neighborhood_size:
                loc[0][1] = sys_args.neighborhood_size
            if loc[0][0] + sys_args.neighborhood_size + 1 >= imgs.shape[0]:
                loc[0][0] = imgs.shape[0] - sys_args.neighborhood_size - 2
            if loc[0][1] + sys_args.neighborhood_size + 1 >= imgs.shape[1]:
                loc[0][1] = imgs.shape[1] - sys_args.neighborhood_size - 2
            locs = locs + loc
            patches_all+=collect_patches(imgs, loc)
            print("patches size: " + str(len(patches_all)))
            # if p == 1:
            #     get_Am(patches_all)

        paramslist, tError = get_LightParams(patches_all)
        t2 = ti.perf_counter()
        print("time: " + str(t2 - t1))

        if bCorrect:

            generate_CorrectionMatrix(paramslist, locs,False)

        else:

            mua3D, mus3D = generate_PatchesCombination_v2(paramslist, locs, imgs.shape, False)
            np.savez(save_tag + "_mua.npz", mua3D)
            np.savez(save_tag + "_mus.npz", mus3D)
            visulize_3Dslices(mua3D)
            visulize_3Dslices(mus3D, btype="mus")

        t3 = ti.perf_counter()
        print("total time: " + str(t3 - t0))

    else:
        mua3D=np.load(save_tag+"_mua.npz")['arr_0']
        mus3D = np.load(save_tag+"_mus.npz")['arr_0']
        visulize_3Dslices(mua3D)
        visulize_3Dslices(mus3D, btype="mus")






