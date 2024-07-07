import os
from scipy.io import loadmat
import numpy as np

dir = os.getcwd() + "/output/"
os.chdir(dir)


files = os.listdir('./')
file_set = set(files)
total_nums = len(files)
cnt = 1
for f in files:
    print('{} / {}'.format(cnt, total_nums))
    cnt += 1
    if  '_H.mci' in f :
        names = f.split('_')
        # if names[0][2:]==names[1]:
        #     continue
        name = '_'.join(names[0:2])
        if '{}_Rd.dat'.format(name) in file_set:
            continue
        #     mcxyz for sample computing, test for lookupmap generation
        os.system('./mcxyz {}'.format(name))
        # os.system('./test {}'.format(name))

