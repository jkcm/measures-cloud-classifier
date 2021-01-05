#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import netCDF4
from netCDF4 import Dataset
import os,datetime,sys,fnmatch
from jdcal import gcal2jd
import math,itertools

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
    Ref1    = MOD02.variables['EV_1KM_RefSB']  
    Ref_050 = np.array(Ref1.reflectance_scales[2]*(Ref1[2,:,:]-Ref1.reflectance_offsets[2]),\
                       dtype='float')/np.cos(np.radians(Solar_Zenith))     # 0.50um at band 10
    Ref_086 = np.array(Ref1.reflectance_scales[10]*(Ref1[10,:,:]-Ref1.reflectance_offsets[10]),\
                       dtype='float')/np.cos(np.radians(Solar_Zenith))     # 0.86um at band 16
    Ref2    = MOD02.variables['EV_500_Aggr1km_RefSB']
    Ref_210 = np.array(Ref2.reflectance_scales[4]*(Ref2[4,:,:]-Ref2.reflectance_offsets[4]),\
                       dtype='float')/np.cos(np.radians(Solar_Zenith))     # 2.10um at band 7   
    print('level-1B reflectance array shape',Ref_210.shape)        
    return lat,lon,CM,CTH,CER,CER_PCL,COT,COT_PCL,Sensor_Zenith,Ref_050,Ref_086,Ref_210

def cloud_fraction(i,j,CM,CTH,Threshold_High,Threshold_Low): 
    TOT_64x64pix = np.sum(np.ravel(CM[50+(i-1)*32:50+(i+1)*32,40+(j-1)*32:40+(j+1)*32])>=0)
    CLH_64x64pix = np.sum(np.ravel(CTH[50+(i-1)*32:50+(i+1)*32, \
                                   40+(j-1)*32:40+(j+1)*32])>=Threshold_High) 
    CLL_64x64pix = np.sum(np.ravel(CTH[50+(i-1)*32:50+(i+1)*32, \
                                   40+(j-1)*32:40+(j+1)*32])<=Threshold_Low)
                    
    high_cf =  np.nan_to_num(CLH_64x64pix)*100.0/TOT_64x64pix   # high cloud fraction [%]
    low_cf  =  np.nan_to_num(CLL_64x64pix)*100.0/TOT_64x64pix   # low cloud fraction     
    return high_cf,low_cf



# 09/21/2018
# Beginning of the program
if __name__ == '__main__':
    
    Satellite = 'TERRA'
    Threshold_High = 4000.0      # Threshold of Cloud_Top_Height for high clouds
    Threshold_Low  = 3000.0      # Threshold of Cloud_Top_Height for low clouds
    
    MOD02_path = '/Users/huasong/Documents/Jupyter/'    
    MOD03_path = '/Users/huasong/Documents/Jupyter/'    
    MOD06_path = '/Users/huasong/Documents/Jupyter/'    
    
    MOD02_fp = 'MOD021KM.A2000060.1400.061.2017171211003.hdf'   # satellite observed radiance
    MOD03_fp = 'MOD03.A2000060.1400.061.2017171200317.hdf'      # lat and lon data
    MOD06_fp = 'MOD06_L2.A2000060.1400.061.2017272135647.hdf'   # cloud retrieval products
    MOD02_fn, MOD03_fn, MOD06_fn =[],[],[]

    for MOD06_flist in  os.listdir(MOD06_path):
        if fnmatch.fnmatch(MOD06_flist, MOD06_fp):
           MOD06_fn = MOD06_flist
    for MOD03_flist in  os.listdir(MOD03_path):
        if fnmatch.fnmatch(MOD03_flist, MOD03_fp):
           MOD03_fn = MOD03_flist
    for MOD02_flist in  os.listdir(MOD02_path):
        if fnmatch.fnmatch(MOD02_flist, MOD02_fp):
           MOD02_fn = MOD02_flist    
    if MOD02_fn and MOD03_fn and MOD06_fn: # if both MOD02, MOD03 and MOD06 products are all available
        
        # Read geolocation,cloud propertity and reflectance data for each granule time
        Lat,Lon,CM,CTH,CER,CER_PCL,COT,COT_PCL,Sensor_Zenith,Ref050,Ref086,Ref210 = \
        read_MODIS_level2_data(MOD06_path+MOD06_fn,MOD03_path+MOD03_fn,MOD02_path+MOD02_fn)
        
        #### Validate data from our previous read file
        #### We pick uo any output filename here                 
        outpath = '/Users/huasong/Documents/Jupyter/outputs/'        
        fout1 =  'IMG_2000060.1400_index_0018_index_0200.npz'
        fout2 =  'MOD_2000060.1400_index_0018_index_0200.npz'

        npzfile1 = np.load(outpath+fout1)
        Ref1_210 = npzfile1['arr_4']    # Reflectance at 2.1um

        npzfile2 = np.load(outpath+fout2)
        CER1 = npzfile2['arr_2']    # CER 
 
        ## To get values from the original MODIS data
        ind_x = 18
        ind_y = 200

        print('Sensor_Zenith = ', Sensor_Zenith[ind_x,ind_y])

        # get 64x64 block mean cloud fraction for this block
        # x = 50+(i-1)*32, so i = 0 for x=18
        # y = 40+(j-1)*32, so j = 6 for y=200
        high_cf,low_cf = cloud_fraction(0,6,CM,CTH,Threshold_High,Threshold_Low)
        print('High_cloud_fraction = ', high_cf)
        print('Low_cloud_fraction = ', low_cf)

        Ref2_210 = Ref210[ind_x:ind_x+64,ind_y:ind_y+64]
        CER2     = CER[ind_x:ind_x+64,ind_y:ind_y+64]

        ### Ref1_210 and CER1 are processed output
        ### Ref2_210 and CER2 are original input
              
        print('Maxmum of Ref1_210 = ', max(max(x) for x in Ref1_210))
        print('Maxmum of Ref2_210 = ', max(max(x) for x in Ref2_210))     
        print('Two data of Reflectance are same: ', (Ref2_210==Ref2_210).all())
    
        print('Maxmum of CER1 = ', max(max(x) for x in CER1))
        print('Maxmum of CER1 = ', max(max(x) for x in CER2))      
        print('Two data of CER are same: ', (CER1==CER2).all())







    
