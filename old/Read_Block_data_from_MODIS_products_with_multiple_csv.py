#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import netCDF4,math,csv
from netCDF4 import Dataset
import os,glob


def find_files_Terra(csvreader):

    Satellite   = 'Terra'         # Satellite name
    Output_path = '/data7/hsong7/Outputs_Chop/Terra/'   # Path for output data
    MOD06_fn,MOD03_fn,MOD02_fn,date_fn = [],[],[],[]    # Initialize input file names and date    

    for row in csvreader:
        if(len(row)>2 and 'MOD06_L2' in row[1] and '.hdf' in row[1]): MOD06_fn.append(row[1])

    n_files = (len(MOD06_fn))     # Total number of MOD06_L2 files
    if(n_files > 0):
      for nf in range(n_files):
          head, tail = os.path.split(MOD06_fn[nf])      # data path is head, filename is tail
          date       = tail.split('.')[1]+'.'+tail.split('.')[2]+'.'+tail.split('.')[3]         
          date_fn.append(date)
          if(glob.glob(head.replace('MOD06_L2','MOD03')+'/MOD03.'+date+'.*.hdf')):
            MOD03_fn.append(glob.glob(head.replace('MOD06_L2','MOD03')+'/MOD03.'+date+'.*.hdf')[0])
          else: MOD03_fn.append('')
          if(glob.glob(head.replace('MOD06_L2','MOD021KM')+'/MOD021KM.'+date+'.*.hdf')):
            MOD02_fn.append(glob.glob(head.replace('MOD06_L2','MOD021KM')+'/MOD021KM.'+date+'.*.hdf')[0])
          else: MOD03_fn.append('')    
    return MOD02_fn,MOD03_fn,MOD06_fn,date_fn,n_files,Output_path


def find_files_Aqua(csvreader):

    Satellite   = 'Aqua'          # Satellite name
    Output_path = '/data7/hsong7/Outputs_Chop/Aqua/'    # Path for output data
    MYD06_fn,MYD03_fn,MYD02_fn,date_fn = [],[],[],[]    # Initialize input file names and date

    for row in csvreader:
        if(len(row)>2 and 'MYD06_L2' in row[1] and '.hdf' in row[1]): MYD06_fn.append(row[1])

    n_files = (len(MYD06_fn))     # Total number of MYD06_L2 files
    if(n_files > 0):
      for nf in range(n_files):
          head, tail = os.path.split(MYD06_fn[nf])      # data path is head, filename is tail
          date       = tail.split('.')[1]+'.'+tail.split('.')[2]+'.'+tail.split('.')[3]
          date_fn.append(date)
          if(glob.glob(head.replace('MYD06_L2','MYD03')+'/MYD03.'+date+'.*.hdf')):
            MYD03_fn.append(glob.glob(head.replace('MYD06_L2','MYD03')+'/MYD03.'+date+'.*.hdf')[0])
          else: MYD03_fn.append('')
          if(glob.glob(head.replace('MYD06_L2','MYD021KM')+'/MYD021KM.'+date+'.*.hdf')):      
            MYD02_fn.append(glob.glob(head.replace('MYD06_L2','MYD021KM')+'/MYD021KM.'+date+'.*.hdf')[0])
          else: MYD02_fn.append('')  
    return MYD02_fn,MYD03_fn,MYD06_fn,date_fn,n_files,Output_path


def read_MODIS_level2_data(MD06_file,MD03_file,MD02_file):

    print('reading the cloud mask from M?D06_L2 product')
    print(MD06_file)    
    MD06  = Dataset(MD06_file, 'r')
    CM1km = MD06.variables['Cloud_Mask_1km']
    CM   = (np.array(CM1km[:,:,0],dtype='byte') & 0b00000110) >>1
    print('level-2 cloud mask array shape',CM.shape)
    CTH1 = MD06.variables['cloud_top_height_1km']
    CTH  = np.array(CTH1[:,:],dtype='float')
    CER1 = MD06.variables['Cloud_Effective_Radius']
    CER  = np.array(CER1[:,:],dtype='float')     
    COT1 = MD06.variables['Cloud_Optical_Thickness']  
    COT  = np.array(COT1[:,:],dtype='float')             
    CER_PCL1 = MD06.variables['Cloud_Effective_Radius_PCL']   
    CER_PCL  = np.array(CER_PCL1[:,:],dtype='float')
    COT_PCL1 = MD06.variables['Cloud_Optical_Thickness_PCL']   
    COT_PCL  = np.array(COT_PCL1[:,:],dtype='float')    
    Fail1    = MD06.variables['Retrieval_Failure_Metric']
    COT_Fail = np.array(Fail1[:,:,0],dtype='float')
    CER_Fail = np.array(Fail1[:,:,1],dtype='float')    

    print('reading the lat-lon from M?D03 product')
    print(MD03_file)
    MD03  = Dataset(MD03_file,'r')
    lat   = MD03.variables['Latitude']
    lon   = MD03.variables['Longitude']
    LM1km = MD03.variables['Land/SeaMask'] 
    LM    = np.array(LM1km[:,:],dtype='i4')     
    Solar_Zenith1  = MD03.variables['SolarZenith']
    Solar_Zenith   = np.array(Solar_Zenith1[:,:],dtype='float')    
    Sensor_Zenith1 = MD03.variables['SensorZenith']
    Sensor_Zenith  = np.array(Sensor_Zenith1[:,:],dtype='float')
    print('level-2 lat-lon array shape',Sensor_Zenith.shape)
    print('maximum(Sensor_Zenith) = ',np.max(Sensor_Zenith))
    
    print('reading the reflectance from M?D02 product')
    print(MD02_file)
    MD02    = Dataset(MD02_file,'r')
    Ref1    = MD02.variables['EV_500_Aggr1km_RefSB']
    Ref_050 = np.array(Ref1.reflectance_scales[1]*(Ref1[1,:,:]-Ref1.reflectance_offsets[1]),dtype='float')       
              # Reflectance at 0.50um, band 4
    Ref_210 = np.array(Ref1.reflectance_scales[4]*(Ref1[4,:,:]-Ref1.reflectance_offsets[4]),dtype='float')       
              # Reflectance at 2.10um, band 7
    Ref2    = MD02.variables['EV_250_Aggr1km_RefSB']
    Ref_086 = np.array(Ref2.reflectance_scales[1]*(Ref2[1,:,:]-Ref2.reflectance_offsets[1]),dtype='float')       
              # Reflectance at 0.86um, band   # /np.cos(np.radians(Solar_Zenith))    
    print('level-1B reflectance array shape',Ref_210.shape)        
    return lat,lon,CM,LM,CTH,CER,CER_PCL,CER_Fail,COT,COT_PCL,COT_Fail,\
           Sensor_Zenith,Ref_050,Ref_086,Ref_210


def Block_info(CM,np_x,np_y):

    ## To define the block numbers along and across each swath
    dim_x  = len(CM[:,0])         # Dimension size of input MODIS data along swath
    dim_y  = len(CM[0,:])         # Dimension size of input MODIS data across swath
    hnp_x  = int(np_x/2)          # Half of the pixel numbers along swath
    hnp_y  = int(np_y/2)          # Half of the pixel numbers across swath
    nb_x   = int(dim_x/hnp_x)-1   # Number of blocks along swath, -1 will lose some pixels along the right bound
    nb_y   = int(dim_y/hnp_y)-1   # Number of blocks across swath, -1 will lose some pixels along the bottom bound
    return nb_x,nb_y,hnp_x,hnp_y


def conditions(Sensor_Zenith,CM,CTH,LM,np_x,np_y):

    # Tunning variables for certain thresholds
    High_cf_max    = 5.0            # The maximum high cloud fraction [%]
    Low_cf_min     = 2.0            # The minimum low cloud fraction [%]
    Land_frac_max  = 10.0           # The maximum land fraction [%]
    Sensor_Zenith_Threshold = 50.0  # Threshold of Sensor Zenith Angle [degree]

    logic = 0.0     # Define the logic for conditioning: 0 means "fail", 1 means "pass"

    # First remove the pixels with sensor zenith angle larger than 50
    # We will start from pixel (0,0), with np_x and np_y increments
    if(0.0 <= Sensor_Zenith[0,0] <= Sensor_Zenith_Threshold and \
       0.0 <= Sensor_Zenith[np_x-1,np_y-1] <= Sensor_Zenith_Threshold):

       # First to get block mean cloud fraction, and land fraction
       high_cf,low_cf = cloud_fraction(CM,CTH)
       land_frac      = land_fraction(LM)
                      
       # Pick our blocks with block-mean high cf < High_cf_max, low cf > Low_cf_min, 
       # high_cf/low_cf < 0.2,and land_frac < Land_frac_max
       if(high_cf < High_cf_max and low_cf > Low_cf_min and \
          high_cf < 0.2*low_cf and land_frac < Land_frac_max):
          logic = 1.0
    return logic  


def cloud_fraction(CM,CTH):

    # Tunning variables for certain thresholds
    Threshold_High = 4000.0         # Threshold of Cloud_Top_Height for high clouds [m]
    Threshold_Low  = 3000.0         # Threshold of Cloud_Top_Height for low clouds  [m]    

    TOT_block = np.sum(np.ravel(CM) >= 0)
    CLH_block = np.sum(np.ravel(CTH) >= Threshold_High) 
    CLL_block = np.sum(np.ravel(CTH) < Threshold_Low) - np.sum(np.ravel(CTH) < 0)                
    high_cf =  np.nan_to_num(CLH_block)*100.0/TOT_block      # High cloud fraction [%]
    low_cf  =  np.nan_to_num(CLL_block)*100.0/TOT_block      # Low cloud fraction  [%]   
    return high_cf,low_cf


def land_fraction(LM):
    LM_block  = np.sum(np.ravel(LM) == 1.0)                 # 1 for land
    land_frac = np.nan_to_num(LM_block)*100.0/LM.size       # Land fraction [%]
    return land_frac


def output_file1(fname,ix,iy,np_x,np_y,Lon,Lat,Ref050,Ref086,Ref210):
               # file name, block x and y cube numbers, reflectance data
    Lon_out  = Lon[ix:(ix+np_x),iy:(iy+np_y)]
    Lat_out  = Lat[ix:(ix+np_x),iy:(iy+np_y)]
    Ref1_out = Ref050[ix:(ix+np_x),iy:(iy+np_y)]
    Ref2_out = Ref086[ix:(ix+np_x),iy:(iy+np_y)]
    Ref3_out = Ref210[ix:(ix+np_x),iy:(iy+np_y)]
    np.savez(fname,Lon_out,Lat_out,Ref1_out,Ref2_out,Ref3_out)

    # Comments on data attributes, (np_x,np_y) 2-D 
    # File description = "Reflectance of 3 bands at blocks (np_x,np_y) for low clouds only"
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


def output_file2(fname,ix,iy,np_x,np_y,Lon,Lat,CER,CER_PCL,CER_Fail,COT,COT_PCL,COT_Fail):
               # file name, block x and y cube numbers, cloud data
    Lon_out      = Lon[ix:(ix+np_x),iy:(iy+np_y)]
    Lat_out      = Lat[ix:(ix+np_x),iy:(iy+np_y)]
    CER_out      = CER[ix:(ix+np_x),iy:(iy+np_y)]
    CER_PCL_out  = CER_PCL[ix:(ix+np_x),iy:(iy+np_y)]
    COT_out      = COT[ix:(ix+np_x),iy:(iy+np_y)]
    COT_PCL_out  = COT_PCL[ix:(ix+np_x),iy:(iy+np_y)]
    CER_Fail_out = CER_Fail[ix:(ix+np_x),iy:(iy+np_y)]
    COT_Fail_out = COT_Fail[ix:(ix+np_x),iy:(iy+np_y)]    
    np.savez(fname,Lon_out,Lat_out,CER_out,CER_PCL_out,CER_Fail_out,COT_out,COT_PCL_out,COT_Fail_out)

    # Comments on the data attributes, (np_x, np_y) 2-D 
    # File description = "CER and COT at blocks (np_x,np_y) for low clouds only"
    # Local attributes to variables
    # Lon_out.units = 'degrees east'
    # Lat_out.units = 'degrees north'
    # CER_out.units = 'micron'
    # CER_out.missing_val = '-9999'
    # CER_out.long_name = 'Cloud_Effective_Radius'
    # CER_PCL_out.units = 'micron'
    # CER_PCL_out.missing_val = '-9999'
    # CER_PCL_out.long_name = 'Cloud_Effective_Radius_PCL'
    # CER_Fail_out.units = 'micron'
    # CER_Fail_out.missing_val = '-9999'
    # CER_Fail_out.long_name = 'Cloud_Effective_Radius_Retrieval_Failure'    
    # COT_out.units = 'none'
    # COT_out.missing_val = '-9999'
    # COT_out.long_name = 'Cloud_Optical_Thickness'
    # COT_PCL_out.units = 'none'
    # COT_PCL_out.missing_val = '-9999'
    # COT_PCL_out.long_name   = 'Cloud_Optical_Thickness_PCL'
    # COT_Fail_out.units = 'none'
    # COT_Fail_out.missing_val = '-9999'
    # COT_Fail_out.long_name = 'Cloud_Optical_Thickness_Retrieval_Failure'
    return


# 10/01/2018
# Beginning of the program
if __name__ == '__main__':
  
  # Tunning variables to define block size
  np_x  = 128           # Block size along swath, please use even number here
  np_y  = 128           # Block size across swath, please use even number here
  
  # CSV files that include the data path and file names
  Input_CSV = ['/home/hsong7/Chop/CSV/LAADS_query.2018-09-24T18_20.csv','/home/hsong7/Chop/CSV/File1_for_test.csv',\
              '/home/hsong7/Chop/CSV/File2_for_test.csv','/home/hsong7/Chop/CSV/File3_for_test.csv',''] 
  CSV_list  = [x for x in Input_CSV if x != '']
  n_csv     = len(CSV_list)
    
  if(n_csv > 0):
    for ncsv in range(n_csv):
       with open(CSV_list[ncsv]) as csvfile:
         print('Input csv file: ',os.path.split(CSV_list[ncsv])[1])
         csvreader    = csv.reader(csvfile,delimiter=',') 
         csv_headings = next(csvreader)
         n_files = 0    # Number of M?D06 files in each csv file
         MD02_fn,MD03_fn,MD06_fn,date_fn,Output_path = [],[],[],[],[]    # Initialize key variables of input data

         if(any('MOD' in string for string in csv_headings)):
            MD02_fn,MD03_fn,MD06_fn,date_fn,n_files,Output_path = find_files_Terra(csvreader)

         if(any('MYD' in string for string in csv_headings)): 
            MD02_fn,MD03_fn,MD06_fn,date_fn,n_files,Output_path = find_files_Aqua(csvreader)

         if(n_files > 0):
           for nf in range(n_files):
              if(MD02_fn[nf] and MD03_fn[nf] and MD06_fn[nf]): 
                # If both MOD02, MOD03 and MOD06 products are all available, we start to chop data
                print('granule time:', date_fn[nf])       
                
                # Read geolocation,cloud propertity and reflectance data for each granule time
                Lat,Lon,CM,LM,CTH,CER,CER_PCL,CER_Fail,COT,COT_PCL,COT_Fail,Sensor_Zenith,Ref050,Ref086,Ref210 = \
                read_MODIS_level2_data(MD06_fn[nf],MD03_fn[nf],MD02_fn[nf])
 
                # Blocks information with size np_x*np_y
                nb_x,nb_y,hnp_x,hnp_y = Block_info(CM,np_x,np_y)

                for i in np.arange(nb_x):     # nb_x blocks along swath
                   for j in np.arange(nb_y):  # nb_y blocks across swath
                       
                       ix = i*hnp_x           # To get index x along swath
                       iy = j*hnp_y           # To get index y across swath

                       # To check whether the block (np_x * np_y size) satisfies the required conditions 
                       logic = conditions(Sensor_Zenith[ix:ix+np_x,iy:iy+np_y],CM[ix:ix+np_x,iy:iy+np_y],\
                                          CTH[ix:ix+np_x,iy:iy+np_y],LM[ix:ix+np_x,iy:iy+np_y],np_x,np_y)

                       if(logic == 1.0):      # If the block satisfies the conditions, chop data to new output files 
                          # To store the values of reflectance, CER and COT to each output file for each block
                          fname1 = 'IMG_'+date_fn[nf]+'_index_'+str(10000+ix)[1:5]+'_index_'+str(10000+iy)[1:5]+ \
                                   '_Block'+str(np_x)+'x'+str(np_y)
                          output = output_file1(Output_path+fname1,ix,iy,np_x,np_y,Lon,Lat,Ref050,Ref086,Ref210)

                          fname2 = 'MOD_'+date_fn[nf]+'_index_'+str(10000+ix)[1:5]+'_index_'+str(10000+iy)[1:5]+ \
                                   '_Block'+str(np_x)+'x'+str(np_y)
                          output = output_file2(Output_path+fname2,ix,iy,np_x,np_y,Lon,Lat,CER,CER_PCL,CER_Fail, \
                                   COT,COT_PCL,COT_Fail)              
             
           print('Total M?D06_L2 file number = ', n_files)
           print('')
    
