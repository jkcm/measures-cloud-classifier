#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import netCDF4
from netCDF4 import Dataset
import os,datetime,sys,fnmatch
from jdcal import gcal2jd
import math
import itertools

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

def output_file1(fname,i,j,Lon,Lat,Ref050,Ref086,Ref210):
               # file name, block x and y cube numbers, reflectance data
    Lon_out  = Lon[50+(i-1)*32:50+(i+1)*32,40+(j-1)*32:40+(j+1)*32]
    Lat_out  = Lat[50+(i-1)*32:50+(i+1)*32,40+(j-1)*32:40+(j+1)*32]
    Ref1_out = Ref050[50+(i-1)*32:50+(i+1)*32,40+(j-1)*32:40+(j+1)*32]
    Ref2_out = Ref086[50+(i-1)*32:50+(i+1)*32,40+(j-1)*32:40+(j+1)*32]
    Ref3_out = Ref210[50+(i-1)*32:50+(i+1)*32,40+(j-1)*32:40+(j+1)*32]
    np.savez(fname,Lon_out,Lat_out,Ref1_out,Ref2_out,Ref3_out)

    # Comments on data attributes, 64x64 2-D 
    # File description = "Reflectance of 3 bands at 64x64 blocks for low clouds only"
    # Local attributes to variables
    # Lon_out.units  = 'degrees east'
    # Lat_out.units  = 'degrees north'
    # Ref1_out.units = 'none'
    # Ref2_out.units = 'none'
    # Ref3_out.units = 'none'
    # Ref1_out.missing_val = '-32767'
    # Ref2_out.missing_val = '-32767'
    # Ref3_out.missing_val = '-32767'
    # Ref1_out.long_name = 'Reflectance at 0.50um'
    # Ref2_out.long_name = 'Reflectance at 0.86um'
    # Ref3_out.long_name = 'Reflectance at 2.10um'

    return

def output_file2(fname,i,j,Lon,Lat,CER,CER_PCL,COT,COT_PCL):
               # file name, block x and y cube numbers, cloud data
    Lon_out = Lon[50+(i-1)*32:50+(i+1)*32,40+(j-1)*32:40+(j+1)*32]
    Lat_out = Lat[50+(i-1)*32:50+(i+1)*32,40+(j-1)*32:40+(j+1)*32]
    CER_out = CER[50+(i-1)*32:50+(i+1)*32,40+(j-1)*32:40+(j+1)*32]
    CER_PCL_out = CER_PCL[50+(i-1)*32:50+(i+1)*32,40+(j-1)*32:40+(j+1)*32]
    COT_out = COT[50+(i-1)*32:50+(i+1)*32,40+(j-1)*32:40+(j+1)*32]
    COT_PCL_out = COT_PCL[50+(i-1)*32:50+(i+1)*32,40+(j-1)*32:40+(j+1)*32]
    np.savez(fname,Lon_out,Lat_out,CER_out,CER_PCL_out,COT_out,COT_PCL_out)

    # Comments on the data attributes, 64x64 2-D 
    # File description = "CER and COT at 64x64 blocks for low clouds only"
    # Local attributes to variables
    # Lon_out.units = 'degrees east'
    # Lat_out.units = 'degrees north'
    # CER_out.units = 'micron'
    # CER_out.missing_val = '-9999'
    # CER_out.long_name = 'Cloud_Effective_Radius'
    # CER_PCL_out.units = 'micron'
    # CER_PCL_out.missing_val = '-9999'
    # CER_PCL_out.long_name = 'Cloud_Effective_Radius_PCL'
    # COT_out.units = 'none'
    # COT_out.missing_val = '-9999'
    # COT_out.long_name = 'Cloud_Optical_Thickness'
    # COT_PCL_out.units = 'none'
    # COT_PCL_out.missing_val = '-9999'
    # COT_PCL_out.long_name   = 'Cloud_Optical_Thickness_PCL'

    return


# 09/21/2018
# Beginning of the program
if __name__ == '__main__':
    
    Satellite      = 'TERRA'
    Threshold_High = 4000.0      # Threshold of Cloud_Top_Height for high clouds [m]
    Threshold_Low  = 3000.0      # Threshold of Cloud_Top_Height for low clouds  [m]
    High_cf_max    = 5.0         # The maximum high cloud fraction [%]
    Low_cf_min     = 1.0         # The maximum high cloud fraction [%]   
    
    MOD02_path = '/Users/huasong/Documents/Jupyter/'    # satellite observed reflectance
    MOD03_path = '/Users/huasong/Documents/Jupyter/'    # lat and lon data
    MOD06_path = '/Users/huasong/Documents/Jupyter/'    # cloud retrieval products
    
    year = [2000]
    days = [31,29,31,30,31,30,31,31,30,31,30,31]
    mn_end=3
    
    for mn in np.arange(1,mn_end):   # mn_end=13 for 12 months from January to December
      dy=np.arange(1,days[mn-1]+1)   # For each month, the number of days are different   
    
      for yr,m,d in  itertools.product(year,[mn],dy):
        date = datetime.datetime(yr,m,d)
        JD01, JD02 = gcal2jd(yr,1,1)
        JD1, JD2 = gcal2jd(yr,m,d)
        JD = np.int((JD2+JD1)-(JD01+JD02) + 1)
        granule_time = datetime.datetime(yr,m,d,0,0)

        while granule_time <= datetime.datetime(yr,m,d,23,55):  # 0-23 hour,0-55 minute
            
           #print('granule time:',granule_time)
           MOD02_fp = 'MOD021KM.A{:04d}{:03d}.{:02d}{:02d}.061.?????????????.hdf'.\
           format(yr,JD,granule_time.hour,granule_time.minute)            
           MOD03_fp = 'MOD03.A{:04d}{:03d}.{:02d}{:02d}.061.?????????????.hdf'.\
           format(yr,JD,granule_time.hour,granule_time.minute)
           MOD06_fp = 'MOD06_L2.A{:04d}{:03d}.{:02d}{:02d}.061.?????????????.hdf'.\
           format(yr,JD,granule_time.hour,granule_time.minute)
            
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
           if MOD02_fn and MOD03_fn and MOD06_fn: 
           # if both MOD02, MOD03 and MOD06 products are all available
             print('granule time:',granule_time)
             # Read geolocation,cloud propertity and reflectance data for each granule time
             Lat,Lon,CM,CTH,CER,CER_PCL,COT,COT_PCL,Sensor_Zenith,Ref050,Ref086,Ref210 = \
             read_MODIS_level2_data(MOD06_path+MOD06_fn,MOD03_path+MOD03_fn,MOD02_path+MOD02_fn)

             # To get 64x64 blocks with high cloud less than 5% and low cloud more than 1%
             for i in np.arange(60):    # 60 cubes along swath
                for j in np.arange(40):  # 40 cubes across swath

                # First remove the pixels with sloar zenith angle larger than 50
                # We will start from pixel (50,40), with -32 to 32 increment
                  if(0.0<= Sensor_Zenith[50+(i-1)*32,40+(j-1)*32] <=50.0):
                    # First to get 64x64 block mean cloud fraction
                    high_cf,low_cf = cloud_fraction(i,j,CM,CTH,Threshold_High,Threshold_Low)

                    # Only keep the block with high cf< High_cf_max and low cf>Low_cf_min
                    if(high_cf < High_cf_max and low_cf > Low_cf_min):
                      x = 50+(i-1)*32 + 10000  # to get 4-digit for index x
                      y = 40+(j-1)*32 + 10000  # to get 4-digit for index y
                      outpath= '/Users/huasong/Documents/Jupyter/outputs/'

                      fname1 = 'IMG_'+MOD03_fp[7:19]+'_index_'+str(x)[1:5]+'_index_'+str(y)[1:5]
                      output = output_file1(outpath+fname1,i,j,Lon,Lat,Ref050,Ref086,Ref210)

                      fname2 = 'MOD_'+MOD03_fp[7:19]+'_index_'+str(x)[1:5]+'_index_'+str(y)[1:5]
                      output = output_file2(outpath+fname2,i,j,Lon,Lat,CER,CER_PCL,COT,COT_PCL)
                    
           granule_time += datetime.timedelta(minutes=5)         
         
    print('Done')
    