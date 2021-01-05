from __future__ import print_function

import matplotlib
matplotlib.use("Agg")

import numpy as np
import netCDF4
from netCDF4 import Dataset
import xarray as xr
import os,datetime,sys,fnmatch
from jdcal import gcal2jd
import math
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def nearest_upsample(data): 
    c = np.empty(tuple(i*2 for i in data.shape), dtype=data.dtype)
    c[0::2,0::2] = data
    c[1::2,0::2] = data
    c[0::2,1::2] = data
    c[1::2,1::2] = data
    return c

def read_MODIS_level2_data(MOD06_file,MOD03_file,MOD02_file):
    print('reading the cloud mask from MOD06_L2 product')
    print(MOD06_file)    
    MOD06 = Dataset(MOD06_file, 'r')
    CM1km = MOD06.variables['Cloud_Mask_1km']
    CM   = (np.array(CM1km[:,:,0],dtype='byte') & 0b00000110) >>1
    print('level-2 cloud mask array shape',CM.shape)
    CTH1 = MOD06.variables['cloud_top_height_1km']
    CTH  = np.array(CTH1[:,:],dtype='float')
    CER1 = MOD06.variables['Cloud_Effective_Radius']
    CER  = np.array(CER1[:,:],dtype='float')     
    COT1 = MOD06.variables['Cloud_Optical_Thickness']  
    COT  = np.array(COT1[:,:],dtype='float')             
    CER_PCL1 = MOD06.variables['Cloud_Effective_Radius_PCL']   
    CER_PCL  = np.array(CER_PCL1[:,:],dtype='float')
    COT_PCL1 = MOD06.variables['Cloud_Optical_Thickness_PCL']   
    COT_PCL  = np.array(COT_PCL1[:,:],dtype='float')    

    print('reading the lat-lon from MYD03 product')
    print(MOD03_file)
    MOD03 = Dataset(MOD03_file,'r')
    lat   = MOD03.variables['Latitude']
    lon   = MOD03.variables['Longitude']
    Solar_Zenith1 = MOD03.variables['SolarZenith']
    Solar_Zenith  = np.array(Solar_Zenith1[:,:],dtype='float')    
    Sensor_Zenith1 = MOD03.variables['SensorZenith']
    Sensor_Zenith  = np.array(Sensor_Zenith1[:,:],dtype='float')
    print('level-2 lat-lon array shape',Sensor_Zenith.shape)
    print('maximum(Sensor_Zenith) = ',np.max(Sensor_Zenith))
    
    print('reading the reflectance from MYD02 product')
    print(MOD02_file)
    MOD02   = Dataset(MOD02_file,'r')
    Ref1    = MOD02.variables['EV_500_RefSB']  
    Ref_band3 = np.array(Ref1.reflectance_scales[0]*(Ref1[0,:,:]-Ref1.reflectance_offsets[0]),\
                       dtype='float')/np.cos(np.radians(nearest_upsample(Solar_Zenith)))     # 0.50um at band 3 (~blue)
    Ref_band4 = np.array(Ref1.reflectance_scales[1]*(Ref1[1,:,:]-Ref1.reflectance_offsets[1]),\
                       dtype='float')/np.cos(np.radians(nearest_upsample(Solar_Zenith)))     # 0.55um at band 4 (~green)
    Ref2    = MOD02.variables['EV_250_Aggr500_RefSB']
    Ref_band1 = np.array(Ref2.reflectance_scales[0]*(Ref2[0,:,:]-Ref2.reflectance_offsets[0]),\
                       dtype='float')/np.cos(np.radians(nearest_upsample(Solar_Zenith)))     # 0.65um at band 1 (~red)
    print('level-1B reflectance array shape',Ref_band1.shape)        
    return lat,lon,CM,CTH,CER,CER_PCL,COT,COT_PCL,Sensor_Zenith,Ref_band1,Ref_band3,Ref_band4

def plot_images(CM,CTH,CER,COT,Ref_band1,Ref_band3,Ref_band4, i, j, modisname):
    savedir = r'/home/disk/p/jkcm/plots/measures/train_plots/it_2'
    modisname = "{}.i{}_j{}.".format(modisname, i, j)
    for n in ['base_with_mask', 'big_context']:
        if not os.path.exists(os.path.join(savedir, n)):
            os.makedirs(os.path.join(savedir, n))
    
    
    rgb = np.transpose(np.array([Ref_band1, Ref_band4, Ref_band3]), axes=(1, 2, 0))
    COT_hr = nearest_upsample(COT)
    CTH_hr = nearest_upsample(CTH)
    CTH_mask = CTH_hr<5000
    imin, imax, jmin, jmax = max(0, i-256), min(rgb.shape[0], i+512), max(0, j-256), min(rgb.shape[1], j+512)
    
    
    fig1, ax1 = plt.subplots();
    ax1.imshow(rgb)
    rect = patches.Rectangle((j,i),256,256,linewidth=1,edgecolor='r',facecolor='none')
    ax1.add_patch(rect)
    plt.tight_layout()
    fig1.savefig(os.path.join(savedir, 'big_context', modisname+'big_context.png'), dpi=300)
    
#     fig3, ax3 = plt.subplots()
#     ax3.imshow(rgb[imin:imax, jmin:jmax, :])
#     rect = patches.Rectangle((255,255),256,256,linewidth=1,edgecolor='r',facecolor='none')
#     ax3.add_patch(rect)
    
#     fig2, ax2 = plt.subplots()
#     ax2.imshow(rgb[i:i+256, j:j+256, :])
    
#     fig4, ax4 = plt.subplots()
#     ax4.imshow(COT[i:i+256, j:j+256])
#     print(rgb.shape)
#     print(CTH_mask[:,:,None].shape)
    
    rgba = (np.concatenate([rgb, CTH_mask[:,:,None]], axis=2))
    fig5, ax5 = plt.subplots(figsize=(6,6))
    ax5.set_facecolor("lightcoral")

    ax5.imshow(rgba[i:i+256, j:j+256, :])
#     ax5.imshow(CTH_mask[i:i+256, j:j+256])
    plt.tight_layout()
    fig5.savefig(os.path.join(savedir, 'base_with_mask', modisname+'base_with_mask.png'), dpi=100)



MOD02_file = '/home/disk/eos4/jkcm/Data/MEASURES/new_modis/MYD02HKM.A2015197.2145.061.2018051030716.hdf'
MOD06_file = r'/home/disk/eos4/jkcm/Data/MEASURES/new_modis/MYD06_L2.A2015197.2145.061.2018051135116.hdf'
MOD03_file = r'/home/disk/eos4/jkcm/Data/MEASURES/new_modis/MYD03.A2015197.2145.061.2018048200537.hdf'

lat,lon,CM,CTH,CER,CER_PCL,COT,COT_PCL,Sensor_Zenith,Ref_band1,Ref_band3,Ref_band4 = read_MODIS_level2_data(MOD06_file,MOD03_file,MOD02_file)

for i in np.arange(0, Ref_band1.shape[0]-256, 128):
    for j in np.arange(0, Ref_band1.shape[1]-256, 128):
        plot_images(CM,CTH,CER,COT,Ref_band1,Ref_band3,Ref_band4, i, j, modisname=os.path.basename(MOD02_file)[:-4])