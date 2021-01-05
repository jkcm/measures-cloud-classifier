import os
import pandas as pd
import requests
import numpy as np
from chop_MODIS_to_blocks128x128_new_conditions import read_MODIS_level2_data, cloud_fraction, land_fraction, plot_images_and_save_retrievals

# Parameters
high_thresh      = 6000         # lower limit for high clouds
low_thresh       = 3500         # upper limit for low clouds
high_cloud_max   = 0.3          # max high cloud fraction allowed
low_cloud_min    = 0.05         # lowest low cloud fraction allowed
max_zenith_angle = 45           # maximum zenith angle allowed
land_frac_max    = 10.0         # The maximum land fraction [%]
np_x             = 128          # Block size along swath, please use even number here
np_y             = np_x          # Block size across swath, please use even number here
np_x_half        = int(np_x/2)       # Half size of the chopped block for outputs
np_y_half        = int(np_y/2)       # Half size of the chopped block for outputs

def get_file_list(date):
    pass

def download_files(file_list):
    #return successful download
    pass

def check_missing_in_folder(folder):
    myd02_list = glob.glob(os.path.join(folder, 'MYD021KM*'))
    myd03_list = glob.glob(os.path.join(folder, 'MYD03*'))
    myd06_list = glob.glob(os.path.join(folder, 'MYD06*'))
    return check_missing(myd02_list, myd03_list, myd06_list)

def check_missing(myd02_list, myd03_list, myd06_list):
    times_02 = ['.'.join(os.path.basename(i).split('.')[1:3]) for i in myd02_list]
    times_03 = ['.'.join(os.path.basename(i).split('.')[1:3]) for i in myd03_list]
    times_06 = ['.'.join(os.path.basename(i).split('.')[1:3]) for i in myd06_list]
    all_times = list(set(times_02+times_03+times_06))
    if len(set([len(x) for x in [all_times, times_02, times_03, times_06]]))==1:
        #nothing missing
        return False
    missing_02, missing_03, missing_06 = [], [], []
    for time in all_times:
        if time not in times_02:
            missing_02.append(time)
        if time not in times_03:
            missing_03.append(time)
        if time not in times_06:
            missing_06.append(time)
    return {"miss_02": missing_02,
            "miss_03": missing_03,
            "miss_06": missing_06}

def convert_csv_to_wget_list(csv_file, savedir=None):
    df = pd.read_csv(csv_file)
    df = df.rename(columns={df.columns[1]: 'filenames'})
    laads_prefix = r'https://ladsweb.modaps.eosdis.nasa.gov'
    all_files = [laads_prefix+i for i in df[df.columns[1]].values]
    if not savedir:
        newfile = csv_file.replace('.csv','_https.txt')
    else:
        newfile = os.path.join(savedir, os.path.basename(csv_file).replace('.csv','_https.txt'))
    with open(newfile, 'w') as f:
        for fname in all_files:
            f.writelines(fname+'\n')
    return all_files


def process_hdf_files(MOD02_file, MOD03_file, MOD06_file, 
                      npz_save_dir, manifest_good, manifest_all, 
                      fig_save_dir=None, plot_and_save_failed_list=None):
    MD02_date = os.path.basename(MOD02_file).split('.')[1]+'.'+os.path.basename(MOD02_file).split('.')[2] #ugh
    try:
        var_dict = read_MODIS_level2_data(MOD06_file,MOD03_file,MOD02_file,MD02_date)
    except Error as e:
        print('Error with reading MODIS 02,03 and 06 files')
        raise e
        return    
    
    save_dir = npz_save_dir
    Figs_dir = fig_save_dir
    manifest1_csv = manifest_good
    manifest2_csv = manifest_all
#     finished_list = NIX this, do it on return from process_hdf_files
    
    
    counter, countern = 0, 0
    for i in np.arange(0, var_dict['Ref_band1'].shape[0]-np_x, np_x):
#         print('\ni = {}'.format(i))
        for j in np.arange(0, var_dict['Ref_band1'].shape[1]-np_y, np_y):
#             print('j = {}'.format(j))
            lc_flag = 0  # 0 if this block does not satisfy conditions, otherwise 1
#             if(len(glob.glob(Data_dir+'*'+MD02_date+'*'+str(10000+i)[1:5]+'*'+str(10000+j)[1:5]+'*.npz'))<1): 
                    # check if this box is already proccessed          
            sensor_zenith = var_dict['Sensor_Zenith'][i+np_x_half, j+np_y_half]
            if(sensor_zenith < max_zenith_angle):           # sensor zenith angle is fine
                    high_cf, low_cf = cloud_fraction(i,j,var_dict['CM'],var_dict['CTH'], \
                                                     high_thresh=high_thresh,low_thresh=low_thresh,np_x=np_x,np_y=np_y)
                    land_frac       = land_fraction(i,j,var_dict['LM'],np_x=np_x,np_y=np_y)
                    if(high_cf < high_cloud_max and low_cf > low_cloud_min and \
                        high_cf < 0.2*low_cf and land_frac < land_frac_max):
#                         print("Condition is satisfied, now plot images and save data for this box")
                        lc_flag = 1
                        fname1  = save_dir + 'IMG_'+os.path.basename(MOD02_file)[:-18]+'_index_'+str(10000+i)[1:5]+'_index_'+ \
                                  str(10000+j)[1:5]+'_Block'+str(np_x)+'x'+str(np_y)
                        fname2  = save_dir + 'MOD_'+os.path.basename(MOD02_file)[:-18]+'_index_'+str(10000+i)[1:5]+'_index_'+ \
                                  str(10000+j)[1:5]+'_Block'+str(np_x)+'x'+str(np_y)
                        try:
                            modis_name,scene_name,context_name = plot_images_and_save_retrievals(var_dict, i, j, \
                                                                 modisname=os.path.basename(MOD02_file)[:-18], savedir=Figs_dir, \
                                                                 np_x=np_x,np_y=np_y,fname1=fname1,fname2=fname2)
                            manifest = pd.DataFrame([{'name': modis_name, 'date': var_dict['date'], \
                                                      'lat': var_dict['lat'][i+np_x_half, j+np_y_half], \
                                                      'lon': var_dict['lon'][i+np_x_half, j+np_y_half],'i':i,'j':j, \
                                                      'sensor_zenith': sensor_zenith, 'low_cf': low_cf, 'high_cf': high_cf, \
                                                      'refl_img': scene_name, 'context_img': context_name}], \
                                                      columns=('name', 'date', 'lat', 'lon', 'i', 'j', 'sensor_zenith', \
                                                      'high_cf', 'low_cf', 'refl_img', 'context_img'))
                            with open(manifest1_csv, 'a') as f: 
                                manifest.to_csv(f, index=False,header=False)
                        except Error as e:
                            raise e
                            print('plot_and_save def is failed')
                            if plot_and_save_failed_list:
                                plot_and_save_failed_files = open(plot_and_save_failed_list,'a')
                                plot_and_save_failed_files.writelines('Filename = '+os.path.basename(MOD02_file) + \
                                                                      ' i = '+str(i)+' j = '+str(j)+'\n')
                                plot_and_save_failed_files.close()
                            else:
                                print('FAILED PLOT AND SAVE: Filename = '+os.path.basename(MOD02_file) + ' i = '+str(i)+' j = '+str(j))
                            continue
                        countern += 1
                        manifest2 = pd.DataFrame([{'name': os.path.basename(MOD02_file), 'n_block': countern, \
                                                 'lat': var_dict['lat'][i+np_x_half, j+np_y_half], \
                                                 'lon': var_dict['lon'][i+np_x_half, j+np_y_half],'i':i,'j':j, \
                                                 'lc_flag': lc_flag}],columns=('name','n_block','lat','lon',\
                                                 'i','j','lc_flag'))
                        with open(manifest2_csv, 'a') as f2: 
                            manifest2.to_csv(f2, index=False,header=False)
                        counter += 1
                        print("Number of chopped boxes = ",counter)
#               finished_files = open(finished_list,'a')
#               finished_files.writelines(MD02_ft+'\n')
#               finished_files.close()
    
    
    

def url_exists(path):
    r = requests.head(path)
    return r.status_code == requests.codes.ok