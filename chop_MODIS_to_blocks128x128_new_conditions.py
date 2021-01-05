# -*- coding: utf-8 -*-
"""
This code is to ingest MODIS 02, 03 and 06_L2 data, make image and output data at each blocks
that satisfies the certain conditions (low clouds, over ocean, and within sensor angle threshold) 
@author: jkcm, hsong, 09/03/2019
"""
from __future__ import print_function, division
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from netCDF4 import Dataset
import os,csv,glob
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import time as systemtime
import numpy.ma as ma


def cloud_fraction(i,j,CM,CTH,high_thresh,low_thresh,np_x,np_y): 
    #clear pixels are defined as where the cloud mask says either uncertain clear, probably clear, or confident clear
    #it is worth noting that a CTH retrieval is still performed where the cloud mask is 'uncertain clear' so we must screen those out.
    #high cloud fraction considers all 'not clear' pixels with a cloud top height > high_thresh, as a fraction of all pixels.
    #low cloud fraction is similar, but uses only (low cloud pixels + clear pixels) in the denominator; this is equivalent to 
    #a random overlap assumption where higher clouds mask the low cloud.
    CTH[CTH<0] = np.nan    # make out missing values -9999
    tot_pix    = CM[i:i+np_x, j:j+np_y].size
    clear_pix_mask = CM[i:i+np_x, j:j+np_y]>1  # 0=confident cloudy, 1=probably cloudy, 2=probably clear, 3=confident clear   
    low_pix = np.nansum(CTH[i:i+np_x, j:j+np_y][~clear_pix_mask]<=low_thresh)  # these are cloudy pixars where CTH is below 3km
    hi_pix  = np.nansum(CTH[i:i+np_x, j:j+np_y][~clear_pix_mask]>=high_thresh) # these are cloudy pixies wthere CTH is above 4km
    clear_pix = np.sum(clear_pix_mask)         # these are the clear pix
    high_cf = hi_pix/tot_pix
    low_cf  = low_pix/(low_pix+clear_pix) 
    return high_cf,low_cf


def cloud_fraction2(i,j,CM,CTH,high_thresh,low_thresh,np_x,np_y): 
    TOT_pix = np.sum(np.ravel(CM[i:i+np_x,j:j+np_y])>=0)
    CLH_pix = np.sum(np.ravel(CTH[i:i+np_x,j:j+np_y]) >=high_thresh) 
    CLL_pix = np.sum(np.ravel(CTH[i:i+np_x,j:j+np_y]) < low_thresh) - \
                     np.sum(np.ravel(CTH[i:i+np_x,j:j+np_y]) < 0)          
    high_cf = np.nan_to_num(CLH_pix)/TOT_pix   # high cloud fraction [%]
    low_cf  = np.nan_to_num(CLL_pix)/TOT_pix   # low cloud fraction  [%]   
    return high_cf,low_cf


def land_fraction(i,j,LM2,np_x,np_y):
    LM         = LM2[i:i+np_x,j:j+np_y]
    LM_block   = np.sum(np.ravel(LM) == 1.0)                 # 1 for land
    land_frac  = np.nan_to_num(LM_block)*100.0/LM.size       # Land fraction [%]
    return land_frac


def read_MODIS_level2_data(MD06_file,MD03_file,MD02_file,date0):
    #Adapted from code written by Hua Song
    print('reading the cloud mask from M?D06_L2 product')
    print(os.path.basename(MD06_file))    
    MD06  = Dataset(MD06_file, 'r')
    CM1km = MD06.variables['Cloud_Mask_1km']
    CM    = (np.array(CM1km[:,:,0],dtype='byte') & 0b00000110) >>1
    print('level-2 cloud mask array shape',CM.shape)
    CTH1 = MD06.variables['cloud_top_height_1km']
    CTH  = np.array(CTH1[:,:],dtype='float')
    #CTH[CTH<-10] = np.nan
    CER1 = MD06.variables['Cloud_Effective_Radius']
    CER  = np.array(CER1[:,:],dtype='float')     
    COT1 = MD06.variables['Cloud_Optical_Thickness']  
    COT  = np.array(COT1[:,:],dtype='float')
    CR1  = MD06.variables['Cirrus_Reflectance']
    CR   = np.array(CR1[:,:],dtype='float')
    #COT[COT<-10] = np.nan
    CER_PCL1 = MD06.variables['Cloud_Effective_Radius_PCL']   
    CER_PCL  = np.array(CER_PCL1[:,:],dtype='float')
    COT_PCL1 = MD06.variables['Cloud_Optical_Thickness_PCL']   
    COT_PCL  = np.array(COT_PCL1[:,:],dtype='float')    
    Fail1    = MD06.variables['Retrieval_Failure_Metric']
    COT_Fail = np.array(Fail1[:,:,0],dtype='float')
    CER_Fail = np.array(Fail1[:,:,1],dtype='float')
    LWP1     = MD06.variables['Cloud_Water_Path']
    LWP      = np.array(LWP1[:,:],dtype='float')
    #LWP[LWP<-10] = np.nan
    MD06.close() 

    print('reading the lat-lon from M?D03 product')
    print(os.path.basename(MD03_file))
    MD03  = Dataset(MD03_file,'r')
    lat   = MD03.variables['Latitude'][:]
    lon   = MD03.variables['Longitude'][:]
    LM1km = MD03.variables['Land/SeaMask']
    LM    = np.array(LM1km[:,:],dtype='i4')
    Solar_Zenith1  = MD03.variables['SolarZenith']
    Solar_Zenith   = np.array(Solar_Zenith1[:,:],dtype='float')    
    Sensor_Zenith1 = MD03.variables['SensorZenith']
    Sensor_Zenith  = np.array(Sensor_Zenith1[:,:],dtype='float')
    print('level-2 lat-lon array shape',Sensor_Zenith.shape)
    print('maximum(Sensor_Zenith) = ',np.max(Sensor_Zenith))
    MD03.close()
    
    print('reading the reflectance from M?D02 product')
    print(os.path.basename(MD02_file))
    MD02  = Dataset(MD02_file,'r')
    Ref1  = MD02.variables['EV_500_Aggr1km_RefSB']
    Ref_band3 = np.array(Ref1.reflectance_scales[0]*(Ref1[0,:,:]-Ref1.reflectance_offsets[0]),\
                         dtype='float')/np.cos(np.radians(Solar_Zenith))     # 0.50um at band 3 (~blue)
    Ref_band4 = np.array(Ref1.reflectance_scales[1]*(Ref1[1,:,:]-Ref1.reflectance_offsets[1]),\
                         dtype='float')/np.cos(np.radians(Solar_Zenith))     # 0.55um at band 4 (~green)
    Ref_band7 = np.array(Ref1.reflectance_scales[4]*(Ref1[4,:,:]-Ref1.reflectance_offsets[4]),\
                         dtype='float')/np.cos(np.radians(Solar_Zenith))     # 2.10um at band 7 
    Ref2      = MD02.variables['EV_250_Aggr1km_RefSB']
    Ref_band1 = np.array(Ref2.reflectance_scales[0]*(Ref2[0,:,:]-Ref2.reflectance_offsets[0]),\
                         dtype='float')/np.cos(np.radians(Solar_Zenith))     # 0.65um at band 1 (~red)
    Ref_band2 = np.array(Ref2.reflectance_scales[1]*(Ref2[1,:,:]-Ref2.reflectance_offsets[1]),\
                         dtype='float')/np.cos(np.radians(Solar_Zenith))     # 0.86um at band 2
    print('level-1B reflectance array shape',Ref_band1.shape)
    MD02.close()

    date = dt.datetime.strptime('.'.join(date0.split('.')[0:2]), 'A%Y%j.%H%M')

    if('MYD' in os.path.split(MD02_file)[1]):        # Do rotation for Aqua data
      lat = lat[::-1,::-1] 
      lon = lon[::-1,::-1]
      CM  = CM[::-1,::-1]
      LM  = LM[::-1,::-1]
      CR  = CR[::-1,::-1]
      CTH = CTH[::-1,::-1]
      CER = CER[::-1,::-1]
      LWP = LWP[::-1,::-1]
      COT = COT[::-1,::-1]
      COT_PCL   = COT_PCL[::-1,::-1]
      CER_PCL   = CER_PCL[::-1,::-1]
      COT_Fail  = COT_Fail[::-1,::-1]
      CER_Fail  = CER_Fail[::-1,::-1]
      Ref_band1 = Ref_band1[::-1,::-1]
      Ref_band4 = Ref_band4[::-1,::-1]
      Ref_band3 = Ref_band3[::-1,::-1]
      Ref_band2 = Ref_band2[::-1,::-1]
      Ref_band7 = Ref_band7[::-1,::-1]
      Solar_Zenith  = Solar_Zenith[::-1,::-1]
      Sensor_Zenith = Sensor_Zenith[::-1,::-1]  
 
    var_dict = {'lat': lat, 'lon': lon, 'CM': CM, 'CTH': CTH, 'CER': CER, 'CER_PCL': CER_PCL,
                'COT': COT, 'COT_PCL': COT_PCL, 'CR': CR, 'LWP': LWP,
                'Sensor_Zenith': Sensor_Zenith, 'Solar_Zenith': Solar_Zenith,
                'Ref_band1': Ref_band1, 'Ref_band4': Ref_band4, 'Ref_band3': Ref_band3,
                'Ref_band2': Ref_band2, 'Ref_band7': Ref_band7, 
                'date': date,'LM': LM,'COT_Fail': COT_Fail,'CER_Fail': CER_Fail}
    return var_dict


def plot_images_and_save_retrievals(var_dict, i, j, modisname, savedir,np_x,np_y,fname1,fname2):
    modisname    = modisname+'_index_'+str(10000+i)[1:5]+'_index_'+str(10000+j)[1:5]
    scene_name   = modisname+'.scene.png'
    context_name = modisname+'.context.png'

### Plot images
    rgb = np.transpose(np.array([np.minimum(var_dict['Ref_band1'], 1), 
                                 np.minimum(var_dict['Ref_band4'], 1), 
                                 np.minimum(var_dict['Ref_band3'], 1)]), axes=(1, 2, 0)) 
### Plot images
    # Scene RGB reflectance image    
    fig1, ax = plt.subplots(figsize=(5,5))
    ax.imshow(rgb[i:i+np_x, j:j+np_y, :])
    ax.axis('off')
    ax.tick_params(axis=u'both', which=u'both',length=0,labelbottom=False,labelleft=False)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.margins(0,0)
    if savedir:
       fig1.savefig(os.path.join(savedir, scene_name), dpi=300, bbox_inches='tight',pad_inches=0)
       plt.close(fig1)
### Plot images

### Save retrievals to outputs
    ## IMG reflectance
    Ref_band1 = var_dict['Ref_band1']
    Ref_band3 = var_dict['Ref_band3']
    Ref_band4 = var_dict['Ref_band4']
    Ref_band2 = var_dict['Ref_band2']
    Ref_band7 = var_dict['Ref_band7']
    Lon_out   = var_dict['lon'][i:(i+np_x),j:(j+np_y)]
    Lat_out   = var_dict['lat'][i:(i+np_x),j:(j+np_y)]
    Ref2_out  = Ref_band2[i:(i+np_x),j:(j+np_y)]
    Ref4_out  = Ref_band4[i:(i+np_x),j:(j+np_y)]
    Ref7_out  = Ref_band7[i:(i+np_x),j:(j+np_y)]
    Ref1_out  = Ref_band1[i:(i+np_x),j:(j+np_y)]
    Ref3_out  = Ref_band3[i:(i+np_x),j:(j+np_y)]
    Lon_out   = ma.filled(Lon_out,np.nan)      # Change the FillVaule to be Nan    
    Lat_out   = ma.filled(Lat_out,np.nan)      # Change the FillVaule to be Nan
    Ref2_out  = ma.filled(Ref2_out,np.nan)     # Change the FillVaule to be Nan 
    Ref4_out  = ma.filled(Ref4_out,np.nan)     # Change the FillVaule to be Nan
    Ref7_out  = ma.filled(Ref7_out,np.nan)     # Change the FillVaule to be Nan 
    Ref1_out  = ma.filled(Ref1_out,np.nan)     # Change the FillVaule to be Nan
    Ref3_out  = ma.filled(Ref3_out,np.nan)     # Change the FillVaule to be Nan 
    np.savez(fname1,**{'Lon':Lon_out,'Lat':Lat_out,'Ref_Band2':Ref2_out,'Ref_Band4':Ref4_out,'Ref_Band7':Ref7_out,\
                       'Ref_Band1':Ref1_out,'Ref_Band3':Ref3_out})

    ## MOD cloud properties
    CTH_out      = var_dict['CTH'][i:(i+np_x),j:(j+np_y)]
    CER_out      = var_dict['CER'][i:(i+np_x),j:(j+np_y)]
    CER_PCL_out  = var_dict['CER_PCL'][i:(i+np_x),j:(j+np_y)]
    COT_out      = var_dict['COT'][i:(i+np_x),j:(j+np_y)]
    COT_PCL_out  = var_dict['COT_PCL'][i:(i+np_x),j:(j+np_y)]
    CER_Fail_out = var_dict['CER_Fail'][i:(i+np_x),j:(j+np_y)]
    COT_Fail_out = var_dict['COT_Fail'][i:(i+np_x),j:(j+np_y)]
    CTH_out  = ma.filled(CTH_out,np.nan)      # Change the FillVaule to be Nan  
    CER_out  = ma.filled(CER_out,np.nan)      # Change the FillVaule to be Nan    
    COT_out  = ma.filled(COT_out,np.nan)      # Change the FillVaule to be Nan
    CER_PCL_out  = ma.filled(CER_PCL_out,np.nan)      # Change the FillVaule to be Nan    
    COT_PCL_out  = ma.filled(COT_PCL_out,np.nan)      # Change the FillVaule to be Nan
    CER_Fail_out = ma.filled(CER_Fail_out,np.nan)     # Change the FillVaule to be Nan    
    COT_Fail_out = ma.filled(COT_Fail_out,np.nan)     # Change the FillVaule to be Nan
    np.savez(fname2,**{'Lon':Lon_out,'Lat':Lat_out,'CTH':CTH_out,'CER':CER_out,'CER_PCL':CER_PCL_out,\
                       'CER_Fail':CER_Fail_out,'COT':COT_out,'COT_PCL':COT_PCL_out,'COT_Fail':COT_Fail_out})
### Save retrievals to outputs
    return (modisname, scene_name, context_name)

def process_date(date):
    pass

# Beginning of the main program
if __name__ == "__main__":

  # The directory of output figures and chopped data
#   Figs_dir0     = r'/att/gpfsfs/briskfs01/ppl/tyuan/group/MODIS_SE_Pacific/blocks_128/figures/'
#   Data_dir0     = r'/att/gpfsfs/briskfs01/ppl/tyuan/group/MODIS_SE_Pacific/blocks_128/data/'

  Figs_dir0 = r'/home/disk/eos4/jkcm/Data/MEASURES/MODIS_downloads/sample/figures/'
  Data_dir0= r'/home/disk/eos4/jkcm/Data/MEASURES/MODIS_downloads/sample/data/' 

  # Define paths for input data 
#   Data_path0    = '/att/gpfsfs/briskfs01/ppl/tyuan/group/MODIS_SE_Pacific/temp/'  # Data path in local server account 

  Data_path0    = '/home/disk/eos4/jkcm/Data/MEASURES/MODIS_downloads/sample/hdf/'

  # Parameters
  high_thresh      = 6000         # lower limit for high clouds
  low_thresh       = 3500         # upper limit for low clouds
  high_cloud_max   = 0.3          # max high cloud fraction allowed
  low_cloud_min    = 0.05         # lowest low cloud fraction allowed
  max_zenith_angle = 45           # maximum zenith angle allowed
  Land_frac_max    = 10.0         # The maximum land fraction [%]
  np_x             = 128          # Block size along swath, please use even number here
  np_y             = 128          # Block size across swath, please use even number here
  np_x_half        = 64           # Half size of the chopped block for outputs
  np_y_half        = 64           # Half size of the chopped block for outputs

#   for year in range(2003,2004):
#     Data_path  = Data_path0 + str(year) + '/'
  for year in ['sample']:
    Data_path  = Data_path0
    MD02_files = sorted(glob.glob(Data_path+'MYD02*.hdf'))
    MD03_files = sorted(glob.glob(Data_path+'MYD03*.hdf'))
    MD06_files = sorted(glob.glob(Data_path+'MYD06*.hdf'))
    MD03_dates = [os.path.basename(MD03_file).split('.')[1]+'.'+os.path.basename(MD03_file).split('.')[2] \
                  for MD03_file in MD03_files]
    MD06_dates = [os.path.basename(MD06_file).split('.')[1]+'.'+os.path.basename(MD06_file).split('.')[2] \
                  for MD06_file in MD06_files]

    if(MD02_files and MD03_files and MD06_files):
      Figs_dir  = Figs_dir0# + str(year) + '/'
      Data_dir  = Data_dir0# + str(year) + '/'
      if not os.path.exists(Figs_dir):os.makedirs(Figs_dir)
      if not os.path.exists(Data_dir):os.makedirs(Data_dir)
      finished_list  = Data_dir + str(year) +'_finished_files_list.txt'
      open(finished_list,'a').close()
      plot_and_save_failed_list  = Data_dir + str(year) + '_plot_and_save_failed_list.txt'
      open(plot_and_save_failed_list,'a').close()
      # Manifest dataframe and csv file
      manifest1_csv = os.path.join(Data_dir, str(year)+'_manifest.csv')
      if(not os.path.exists(manifest1_csv)):
        header1 = pd.DataFrame(columns=('name', 'date', 'lat', 'lon', 'i', 'j', 'sensor_zenith', \
                                        'high_cf', 'low_cf', 'refl_img', 'context_img'))
        header1.to_csv(manifest1_csv, index=False)
      manifest2_csv = os.path.join(Data_dir, str(year)+'_manifest_allfiles.csv')
      if(not os.path.exists(manifest2_csv)):
        header2 = pd.DataFrame(columns=('name', 'n_block', 'lat', 'lon', 'i', 'j', 'lc_flag'))
        header2.to_csv(manifest2_csv, index=False)

      for MD02_ft in MD02_files:
          try:
             MD02_date = os.path.basename(MD02_ft).split('.')[1]+'.'+os.path.basename(MD02_ft).split('.')[2]
             MD03_ft   = MD03_files[MD03_dates.index(MD02_date)]
             MD06_ft   = MD06_files[MD06_dates.index(MD02_date)]
          except:
             print('No matching MODIS 02,03 and 06 files found')
             continue

          if(MD02_ft in open(finished_list).read()):
            print(MD02_ft)
            print('This gradule file is already processed, so skip it\n')
          elif(os.stat(MD06_ft).st_size>0 and os.stat(MD03_ft).st_size>0 and os.stat(MD02_ft).st_size>0):
              try:
                 var_dict = read_MODIS_level2_data(MD06_ft,MD03_ft,MD02_ft,MD02_date)
              except:
                 print('Error with reading MODIS 02,03 and 06 files')
                 continue
              counter, countern = 0, 0
              for i in np.arange(0, var_dict['Ref_band1'].shape[0]-np_x, np_x):
                print('\ni = {}'.format(i))
                for j in np.arange(0, var_dict['Ref_band1'].shape[1]-np_y, np_y):
                  print('j = {}'.format(j))
                  lc_flag = 0  # 0 if this block does not satisfy conditions, otherwise 1
                  if(len(glob.glob(Data_dir+'*'+MD02_date+'*'+str(10000+i)[1:5]+'*'+str(10000+j)[1:5]+'*.npz'))<1): 
                    # check if this box is already proccessed          
                    sensor_zenith = var_dict['Sensor_Zenith'][i+np_x_half, j+np_y_half]
                    if(sensor_zenith < max_zenith_angle):           # sensor zenith angle is fine
                      high_cf, low_cf = cloud_fraction(i,j,var_dict['CM'],var_dict['CTH'], \
                                                       high_thresh=high_thresh,low_thresh=low_thresh,np_x=np_x,np_y=np_y)
                      land_frac       = land_fraction(i,j,var_dict['LM'],np_x=np_x,np_y=np_y)
                      if(high_cf < high_cloud_max and low_cf > low_cloud_min and \
                        high_cf < 0.2*low_cf and land_frac < Land_frac_max):
                        print("Condition is satisfied, now plot images and save data for this box")
                        lc_flag = 1
                        fname1  = Data_dir + 'IMG_'+os.path.basename(MD02_ft)[:-18]+'_index_'+str(10000+i)[1:5]+'_index_'+ \
                                  str(10000+j)[1:5]+'_Block'+str(np_x)+'x'+str(np_y)
                        fname2  = Data_dir + 'MOD_'+os.path.basename(MD02_ft)[:-18]+'_index_'+str(10000+i)[1:5]+'_index_'+ \
                                  str(10000+j)[1:5]+'_Block'+str(np_x)+'x'+str(np_y)
                        try:
                          modis_name,scene_name,context_name = plot_images_and_save_retrievals(var_dict, i, j, \
                                                               modisname=os.path.basename(MD02_ft)[:-18], savedir=Figs_dir, \
                                                               np_x=np_x,np_y=np_y,fname1=fname1,fname2=fname2)
                          manifest = pd.DataFrame([{'name': modis_name, 'date': var_dict['date'], \
                                                    'lat': var_dict['lat'][i+np_x_half, j+np_y_half], \
                                                    'lon': var_dict['lon'][i+np_x_half, j+np_y_half],'i':i,'j':j, \
                                                    'sensor_zenith': sensor_zenith, 'low_cf': low_cf, 'high_cf': high_cf, \
                                                    'refl_img': scene_name, 'context_img': context_name}], \
                                                    columns=('name', 'date', 'lat', 'lon', 'i', 'j', 'sensor_zenith', \
                                                    'high_cf', 'low_cf', 'refl_img', 'context_img'))
                          with open(manifest1_csv, 'a') as f: manifest.to_csv(f, index=False,header=False)
                        except:
                          print('plot_and_save def is failed')
                          plot_and_save_failed_files = open(plot_and_save_failed_list,'a')
                          plot_and_save_failed_files.writelines('Filename = '+os.path.basename(MD02_ft) + \
                                                                ' i = '+str(i)+' j = '+str(j)+'\n')
                          plot_and_save_failed_files.close()
                          continue
                      countern += 1
                      manifest2 = pd.DataFrame([{'name': os.path.basename(MD02_ft), 'n_block': countern, \
                                                 'lat': var_dict['lat'][i+np_x_half, j+np_y_half], \
                                                 'lon': var_dict['lon'][i+np_x_half, j+np_y_half],'i':i,'j':j, \
                                                 'lc_flag': lc_flag}],columns=('name','n_block','lat','lon',\
                                                 'i','j','lc_flag'))
                      with open(manifest2_csv, 'a') as f2: manifest2.to_csv(f2, index=False,header=False)
                  counter += 1
                  print("Number of chopped boxes = ",counter)
              finished_files = open(finished_list,'a')
              finished_files.writelines(MD02_ft+'\n')
              finished_files.close()