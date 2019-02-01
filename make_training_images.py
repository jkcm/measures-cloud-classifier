# -*- coding: utf-8 -*-
"""
@author: jkcm
"""
from __future__ import print_function, division
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from netCDF4 import Dataset
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime as dt
from matplotlib import gridspec
import pandas as pd

def nearest_upsample(data):
    #just some quick code to reshape the 1km resolution data to 500m. 
    #halving indices etc would also work, but the marginal reduction in code efficiency isn't worth the cost to legibility
    c = np.empty(tuple(i*2 for i in data.shape), dtype=data.dtype)
    c[0::2,0::2] = data
    c[1::2,0::2] = data
    c[0::2,1::2] = data
    c[1::2,1::2] = data
    return c

def cloud_fraction(i,j,CM,CTH,high_thresh=4000,low_thresh=3000): 
    #clear pixels are defined as where the cloud mask says either uncertain clear, probably clear, or confident clear
    #it is worth noting that a CTH retrieval is still performed where the cloud mask is 'uncertain clear' so we must screen those out.
    #high cloud fraction considers all 'not clear' pixels with a cloud top height > high_thresh, as a fraction of all pixels.
    #low cloud fraction is similar, but uses only (low cloud pixels + clear pixels) in the denominator; this is equivalent to 
    #a random overlap assumption where higher clouds mask the low cloud.
    tot_pix = CM[i:i+128, j:j+128].size
    clear_pix_mask = CM[i:i+128, j:j+128]>0
    low_pix = np.sum(CTH[i:i+128, j:j+128][~clear_pix_mask]<=low_thresh) # these are cloudy pixars where CTH is below 3km
    hi_pix = np.sum(CTH[i:i+128, j:j+128][~clear_pix_mask]>=high_thresh) # these are cloudy pixies wthere CTH is above 4km
    clear_pix = np.sum(clear_pix_mask) #these are the clear pix
    high_cf = hi_pix/tot_pix
    low_cf = low_pix/(low_pix+clear_pix)  
    return high_cf,low_cf

def read_MODIS_level2_data(MOD06_file,MOD03_file,MOD02_file):
    #Adapted from code written by Hua Song
    print('reading the cloud mask from MOD06_L2 product')
    print(MOD06_file)    
    MOD06 = Dataset(MOD06_file, 'r')
    CM1km = MOD06.variables['Cloud_Mask_1km']
    CM   = (np.array(CM1km[:,:,0],dtype='byte') & 0b00000110) >>1
    print('level-2 cloud mask array shape',CM.shape)
    CTH1 = MOD06.variables['cloud_top_height_1km']
    CTH  = np.array(CTH1[:,:],dtype='float')
    CTH[CTH<-10] = np.nan
    CER1 = MOD06.variables['Cloud_Effective_Radius']
    CER  = np.array(CER1[:,:],dtype='float')     
    COT1 = MOD06.variables['Cloud_Optical_Thickness']  
    COT  = np.array(COT1[:,:],dtype='float')*COT1.scale_factor
    CR1 = MOD06.variables['Cirrus_Reflectance']
    CR  = np.array(CR1[:,:],dtype='float')*CR1.scale_factor
    COT[COT<-10] = np.nan
    CER_PCL1 = MOD06.variables['Cloud_Effective_Radius_PCL']   
    CER_PCL  = np.array(CER_PCL1[:,:],dtype='float')
    COT_PCL1 = MOD06.variables['Cloud_Optical_Thickness_PCL']   
    COT_PCL  = np.array(COT_PCL1[:,:],dtype='float')    
    LWP1 = MOD06.variables['Cloud_Water_Path']  
    LWP  = np.array(LWP1[:,:],dtype='float')*LWP1.scale_factor
    LWP[LWP<-10] = np.nan

    print('reading the lat-lon from MYD03 product')
    print(MOD03_file)
    MOD03 = Dataset(MOD03_file,'r')
    lat   = MOD03.variables['Latitude'][:]
    lon   = MOD03.variables['Longitude'][:]
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
    
    date = dt.datetime.strptime('.'.join(MOD02_file.split('.')[-5:-3]), 'A%Y%j.%H%M')

    var_dict = {'lat': lat, 'lon': lon, 'CM': CM, 'CTH': CTH, 'CER': CER, 'CER_PCL': CER_PCL,
                'COT': COT, 'COT_PCL': COT_PCL, 'CR': CR, 'LWP': LWP,
                'Sensor_Zenith': Sensor_Zenith, 'Solar_Zenith': Solar_Zenith,
                'Ref_band1': Ref_band1, 'Ref_band4': Ref_band4, 'Ref_band3': Ref_band3,
                'date': date}
    return var_dict

def plot_images(var_dict, i, j, modisname, savedir):
    modisname = "{}.i{}_j{}".format(modisname, i, j)
    if savedir:
        if not os.path.exists(savedir):
                os.makedirs(savedir)
            
    rgb = np.transpose(np.array([np.minimum(var_dict['Ref_band1'], 1), 
                                 np.minimum(var_dict['Ref_band1'], 1), 
                                 np.minimum(var_dict['Ref_band1'], 1)]), axes=(1, 2, 0))
    COT_hr = nearest_upsample(var_dict['COT'])
    COT_PCL_hr = nearest_upsample(var_dict['COT_PCL'])
    CTH_hr = nearest_upsample(var_dict['CTH'])
    LWP_hr = nearest_upsample(var_dict['LWP'])
    CR_hr = nearest_upsample(var_dict['CR'])
    sen_zen = nearest_upsample(var_dict['Sensor_Zenith'])
    CTH_mask = CTH_hr<5000
    
    # Scene RGB reflectance image
    fig1, ax = plt.subplots(figsize=(5,5))
    ax.imshow(rgb[i:i+128, j:j+128, :])
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)    
    ax.margins(0,0)    
    scene_name = modisname+'.scene.png'
    if savedir:
        fig1.savefig(os.path.join(savedir, scene_name), dpi=300)
        plt.close(fig1)

    #context collage
    fig2 = plt.figure(figsize=(7.5,6.5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[3, 2, 2])
    gs.update(left=0.00, right=1, wspace=0.05, hspace=0.01)
    ax1 = plt.subplot(gs[:, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[1, 1])
    ax5 = plt.subplot(gs[1, 2])
  
    ## granule
    ax1.imshow(rgb)
    rect = patches.Rectangle((j,i),128,128,linewidth=1,edgecolor='r',facecolor='none')
    ax1.add_patch(rect)
    ax1.axis('off')
    ## context
    imin, imax, jmin, jmax = max(0, i-128), min(rgb.shape[0], i+256), max(0, j-128), min(rgb.shape[1], j+256)
    ax2.imshow(rgb[imin:imax, jmin:jmax, :])
    rect = patches.Rectangle((j-jmin, i-imin),128,128,linewidth=1,edgecolor='r',facecolor='none')
    ax2.add_patch(rect)
    ax2.axis('off')
    ## scene
    ax3.imshow(rgb[i:i+128, j:j+128, :])
    ax3.axis('off')    
    ## Scene CTH image
    cth = ax4.imshow(CTH_hr[i:i+128, j:j+128]/1000)
    ax4.axis('off')
    cb = plt.colorbar(cth, ax=ax4, orientation='horizontal', pad=0.01)
    cb.set_label('CTH (km)', size=16)
    for tick in cb.ax.get_xticklabels():
        tick.set_rotation(45)
        
    ## Scene LWP image
    lwp = ax5.imshow(LWP_hr[i:i+128, j:j+128]/1000)
    ax5.axis('off')
    cb = plt.colorbar(lwp, ax=ax5, orientation='horizontal', pad=0.01)
    cb.set_label('LWP (g/m2)', size=16)
    for tick in cb.ax.get_xticklabels():
        tick.set_rotation(45)
    context_name = modisname+'.context.png'
    if savedir:
        fig2.savefig(os.path.join(savedir, context_name), dpi=100, bbox_inches='tight')
        plt.close(fig2)

    return (modisname, scene_name, context_name)

if __name__ == "__main__":
    #adjust appropriately 
    MOD02_files = [r'/home/disk/eos4/jkcm/Data/MEASURES/new_modis/MYD02HKM.A2015197.2145.061.2018051030716.hdf']
    MOD06_files = [r'/home/disk/eos4/jkcm/Data/MEASURES/new_modis/MYD06_L2.A2015197.2145.061.2018051135116.hdf']
    MOD03_files = [r'/home/disk/eos4/jkcm/Data/MEASURES/new_modis/MYD03.A2015197.2145.061.2018048200537.hdf']
    savedir = r'/home/disk/p/jkcm/plots/measures/train_plots/it_11'

    #params
    high_thresh = 4000 #lower limit for high clouds
    low_thresh = 3000 #upper limit for low clouds
    high_cloud_max = 0.5 # max high cloud fraction allowed
    low_cloud_min = 0.1 #lowest low cloud fraction allowed
    max_zenith_angle = 45 # maximum zenith angle allowed
    
    #manifest dataframe
    manifest = pd.DataFrame(columns=('name', 'date', 'lat', 'lon', 'i', 'j', 'sensor_zenith', 'high_cf', 'low_cf', 'refl_img', 'context_img'))
    
    for MOD06_file,MOD03_file,MOD02_file in zip(MOD06_files,MOD03_files,MOD02_files):

        # just making sure that all products are using the same granule
        dates = ['.'.join(f.split('.')[-5:-3]) for f in [MOD06_file,MOD03_file,MOD02_file]]
        print('working on ' + dates[1])
        assert dates[0] == dates[1] and dates[1] == dates[2]
    
        var_dict = read_MODIS_level2_data(MOD06_file,MOD03_file,MOD02_file)
        counter = 0
        for i in np.arange(0, var_dict['Ref_band1'].shape[0]-128, 128):
            print('i={}'.format(i))
            for j in np.arange(0, var_dict['Ref_band1'].shape[1]-128, 128):
                print('j={}'.format(j))
                sensor_zenith = nearest_upsample(var_dict['Sensor_Zenith'])[i+64, j+64]
                if sensor_zenith < max_zenith_angle: # sensor zenith angle is fine
                    high_cf, low_cf = cloud_fraction(i,j,nearest_upsample(var_dict['CM']),nearest_upsample(var_dict['CTH']),high_thresh=high_thresh,low_thresh=low_thresh)
                    if high_cf < high_cloud_max and low_cf > low_cloud_min:  # not too much high cloud, not too little low cloud
                        modis_name, scene_name, context_name = plot_images(var_dict, i, j, modisname=os.path.basename(MOD02_file)[:-4], savedir=savedir)
                        manifest = manifest.append([{'name': modis_name, 'date': var_dict['date'], 
                                           'lat': nearest_upsample(var_dict['lat'])[i+64, j+64] ,'lon': nearest_upsample(var_dict['lon'])[i+64, j+64], 'i':i, 'j':j,
                                           'sensor_zenith': sensor_zenith, 'low_cf': low_cf, 'high_cf': high_cf,
                                           'refl_img': scene_name, 'context_img': context_name}])

                    counter+= 1
                    print(counter)
            manifest.to_csv(os.path.join(savedir, 'manifest.csv'), index=False)
