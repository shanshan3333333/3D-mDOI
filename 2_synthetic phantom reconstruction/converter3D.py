import numpy as np
from PIL import Image

save_tag = "pigment"
neighbouring = 18
mus = np.load(save_tag + "_mua.npz")['arr_0'][neighbouring:-neighbouring,neighbouring:-neighbouring,:]


vmax = np.max(mus)
vmin = np.min(mus)


# some might need the other normalization method listed below for better visualization
# mua = abs(1 - mua) * 4095#
mua = (mua-vmin) * 4095/(vmax-vmin)
mua.astype(np.uint16)
name = "{}_{}_{}.tif"

for i in range(mua.shape[2]):
    im = Image.fromarray(mua[:,:,i])
    file_name=name.format(save_tag,'_mua',str(i))
    im.save(file_name)
a=1