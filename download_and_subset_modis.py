import xarray as xr
from urllib.request import urlopen
from urllib.error import HTTPError
import re
from LoopTimer import LoopTimer
import os
import datetime as dt
import multiprocessing as mp
import argparse
from functools import partial
from itertools import repeat
from random import randint
from time import sleep, asctime

def get_filelist_from_day_url(day_url, prefix, tries=10):
    c=0
    mostrecenterror = IOError("something has gone horribly and mysteriously wrong!")
    while c<tries:
#         print('opening {}, try {}'.format(day_url, c+1))
        try:
            urlpath =urlopen(day_url)
            string = urlpath.read().decode('utf-8')
            pattern = re.compile('{}\..*\.hdf'.format(prefix)) #the pattern actually creates duplicates in the list
            filelist = sorted(list(set(pattern.findall(string))))
            print('got file list! {} files.'.format(len(filelist)))
            return filelist
        except HTTPError as e:
            print('HTTPError, try {}: {}'.format(c, e))
            mostrecenterror = e
            c+=1
            continue
        except OSError as e:
            print('OSError, try {}: {}'.format(c, e))
            mostrecenterror = e
            c+=1
            continue                
    print('count not get file list, max tries exceeded ({})'.format(day_url))
    raise mostrecenterror
            
            
def subset_mod02_dataset(dataset):
    band_01 = dataset['EV_250_Aggr1km_RefSB'].sel(Band_250M=1.0).drop('Band_250M').rename('band_01')
    band_02 = dataset['EV_250_Aggr1km_RefSB'].sel(Band_250M=2.0).drop('Band_250M').rename('band_02')
    band_03 = dataset['EV_500_Aggr1km_RefSB'].sel(Band_500M=3.0).drop('Band_500M').rename('band_03')
    band_04 = dataset['EV_500_Aggr1km_RefSB'].sel(Band_500M=4.0).drop('Band_500M').rename('band_04')
    band_07 = dataset['EV_500_Aggr1km_RefSB'].sel(Band_500M=7.0).drop('Band_500M').rename('band_07')
    band_26 = dataset['EV_1KM_RefSB'].sel(Band_1KM_RefSB=26.0).drop('Band_1KM_RefSB').rename('band_26')
    band_20 = dataset['EV_1KM_Emissive'].sel(Band_1KM_Emissive=20.0).drop('Band_1KM_Emissive').rename('band_20')
    band_31 = dataset['EV_1KM_Emissive'].sel(Band_1KM_Emissive=31.0).drop('Band_1KM_Emissive').rename('band_31')
    band_32 = dataset['EV_1KM_Emissive'].sel(Band_1KM_Emissive=32.0).drop('Band_1KM_Emissive').rename('band_32')
    sensor_z = dataset['SensorZenith'].rename('SensorZenith')
    solar_z = dataset['SensorZenith'].rename('SolarZenith')
    gflags = dataset['gflags'].rename('gflags')
    all_vars = [band_01, band_02, band_03, band_04, band_07, band_26, band_20, band_31, band_32, sensor_z, solar_z, gflags]
    all_vars_dict = {i.name: i for i in all_vars}
    new_ds = xr.Dataset(all_vars_dict)
    return new_ds

def subset_and_save(dap_dataset, save_loc, subset_fn):
    print('subsetting and saving to {}'.format(save_loc))
    ds = subset_fn(dap_dataset)
    ds.to_netcdf(save_loc)
    return
    
    
def save_filelist_to_dir(filelist, savedir, day_url, subset_fn, modis_prefix):
    downloaded_granules = os.listdir(savedir)
    nightfile = os.path.join(savedir, 'skipped_granules')
    if os.path.exists(nightfile):
        with open(nightfile, 'r+') as f:
            downloaded_granules.extend(f.read().splitlines())
#     lt = LoopTimer(len(filelist))
    for gran in filelist:
#         lt.update()
        savename = os.path.join(savedir, gran[:-3]+'subset.nc')
        if os.path.basename(savename) in downloaded_granules or gran in downloaded_granules:
            print('already downloaded {}'.format(gran))
            continue
        c=0
        while c<10:
            print('opening {} for download, try {}'.format(gran, c+1))
            try:
                with xr.open_dataset(day_url+gran) as dataset:
                    if modis_prefix == 'MYD06_L2': # no night mode filter
                        subset_and_save(dataset, savename, subset_fn)
                        downloaded_granules.append(gran)
                        print('downloaded {}'.format(gran))
                    elif modis_prefix =='MYD021KM':
                        if dataset.attrs["Number_of_Night_mode_scans"] == 0:
                            subset_and_save(dataset, savename, subset_fn)
                            downloaded_granules.append(gran)
                            print('downloaded {}'.format(gran))
                        else:
                            print('nighttime, skipping')
                            with open(nightfile, 'a+') as f:
                                f.write(gran+'\n')
                print('{} successfully saved!'.format(gran))
                break
            except OSError as e:
                print('OSError, try {}: {}'.format(c, e))
                c+=1
                continue
            except KeyError as e:
                print('KeyError on {}, skipping'.format(gran))
                with open(nightfile, 'a+') as f:
                    f.write(gran+'\n')
                break
            except RuntimeError as e:
                print("Runtime Error on {}, ignoring for now".format(gran))
#                 raise e

    return

def download_from_date(date, modis_prefix):
    day_url = r'https://ladsweb.modaps.eosdis.nasa.gov/opendap/allData/61/{1}/{0:%Y}/{0:%j}/'.format(date, modis_prefix)
    savedir = r'/home/disk/eos9/jkcm/Data/modis/{1}/{0:%Y}/{0:%j}/'.format(date, modis_prefix)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
#     print('saving data to {}'.format(savedir))
    sleep(randint(0,100)/100.)
    filelist = get_filelist_from_day_url(day_url, modis_prefix)
    print('file list downloaded: {} total possible files'.format(len(filelist)))
    
    if modis_prefix=='MYD021KM':
        subset_fn = subset_mod02_dataset
    elif modis_prefix in ['MYD06_L2', 'MYD03']:
        subset_fn = lambda x: x.copy(deep=True) #no subsetting, just return copy of dataset
        
    save_filelist_to_dir(filelist, savedir, day_url, subset_fn, modis_prefix)
    return 0

                
if __name__ == "__main__":
    print('\n\n\n\nbeginning: {}'.format(asctime()))
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix', '-p', help='specify the exact MODIS product prefix to download')
    parser.add_argument('--start', '-s', help='specify the start date of the download period (YYYYMMDD)')
    parser.add_argument('--end', '-e', help='specify the start date of the  modis period (YYYYMMDD)')
    args=parser.parse_args()
    if args.start:
        start = dt.datetime.strptime(args.start, '%Y%m%d')
    else:
        start = dt.datetime(2015, 7, 1)
    if args.end:
        end = dt.datetime.strptime(args.end, '%Y%m%d')
    else:
        end = dt.datetime(2016, 9, 30)
    if args.prefix:
        prefix = args.prefix
        print('prefix: "'+prefix+ '"')
    else:
        prefix = 'MYD021KM'
        
    print('Downloading all {} daytime data from {:%Y/%m/%d} to {:%Y/%m/%d}. Using {} processors\n\n\n\n.'.format(
        prefix, start, end, mp.cpu_count()))

    
    ndays = int((end-start).total_seconds()/(60*60*24)+1)
    dates = [start + dt.timedelta(days=i) for i in range(ndays)]
    
    def prefixed_func(date):
#         print('working on {}: starting at {}'.format(date, asctime()))
        print('working on {}'.format(date))
        download_from_date(date, prefix)
        
    pool = mp.Pool(mp.cpu_count())
#     results = pool.starmap(download_from_date, zip(dates, repeat(prefix)))
    results = pool.map(prefixed_func, dates)