from __future__ import print_function
import numpy as np
import math
import os,datetime,sys,fnmatch
import glob
import matplotlib.pyplot as plt

def make_image(data, outputname, size=(1, 1), dpi=100):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, aspect='equal',cmap ='gray')
    plt.savefig(outputname, dpi=dpi)
    plt.close()

if __name__ == '__main__':
    dir = #npzfile path
    files = glob.glob(dir+'IMG*.npz')
    sample_size = 50000l
    idx = np.random.choice(len(files),sample_size,replace =False)
    for i in range(len(idx)):
        npzfile = np.load(files[idx[i]])
        temp_img = npzfile['arr_2']
        img_name = files[idx[i]].replace('blocks_128','train/samples')
        img_name = img_name.replace('.npz','.png')
        make_image(temp_img, img_name, size=(1, 1), dpi=100)