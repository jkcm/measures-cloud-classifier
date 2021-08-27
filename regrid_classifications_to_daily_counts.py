#Code for regridding MEASURES low cloud classifications to a regular dataset
#adapted from Ryan Eastman's MATLAB code for Python

import numpy as np
from scipy.interpolate import interp2d
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
import datetime
import xarray as xr
print('imported!')

def dist2lat(dx,dy,lat,lon):
    """[dlat,dlon]=dist2lat(dx,dy,lat,lon): in ~/EPIC/scripts/
    #computes change in lat, lon resulting from a change in cartesian
    #distance dx, dy (in m) originating from (lat,lon).
    #
    #SEE ALSO: lat2dist
    #adapted from Ryan Eastman's MATLAB code for Python
    """
    #CHANGE IN LAT:
    #======================================
    #NOTE:   d(lat. arc)       360 deg
    #      -------------- = ---------------
    #      dist traversed    2*pi*rad. earth

        dlat=360*dy/(2*np.pi*6.37e6)

    #CHANGE IN LON:
    #======================================
    #NOTE:   d(lon. arc)       360 deg
    #      -------------- = ---------------
    #      dist traversed    2*pi*rad. lat. circle

        dlon=360*dx/(2*np.pi*6.37e6*np.cos(np.pi/180*lat));

    return (dlat, dlon)

#reading in Tianle's files:
ds = xr.open_dataset('/home/disk/eos4/jkcm/Data/MEASURES/beta_data/atlantic/2015_NH_Atlantic_model_predictions_of_cloud_types.nc')
ds = ds.drop([i for i in ds.data_vars if 'num_blocks_obs' in ds[i].dims]) # dropping all the non-classified blocks

# establish grid
gridres = 1;
latgrid = np.arange(75-gridres/2,-75,-gridres)
longrid = np.arange(-180+gridres/2,180,1)
[LONS, LATS] = np.meshgrid(longrid,latgrid)
mcctypes = [0,1,2,3,4,5]

target_dir = '/home/disk/eos4/jkcm/Data/MEASURES/regridded_MCC'

dayofyear = np.array([int(i[14:17]) for i in ds.block_low.values])
year = np.array([int(i[10:14]) for i in ds.block_low.values])

for yr in np.unique(year):
    

    ydays = 365 + int(pd.Timestamp(yr, 12, 31).is_leap_year)
    for day in range(1, ydays+1):
        print(f'working on {yr, day}...')
    
        #set up xarray dataset for this day        
        save_ds_attrs = {
            'Contact': 'Johannes Mohrmann (jkcm@uw.edu)',
            'Institution': 'University of Washington, Dept. of Atmospheric Sciences',
            'Creation Time': str(datetime.datetime.utcnow()),
            'Project': 'UW-MEASURES cloud classifier',
            'website': 'https://github.com/jkcm/measures-cloud-classifier',
            'References': 'For classification scheme: https://doi.org/10.5194/amt-13-6989-2020',
            'authors': 'Johannes Mohrmann (data, adapted code), Tianle Yuan (raw data), Ryan Eastman (original regridding code)'}
        
        save_ds = xr.Dataset(attrs=save_ds_attrs)
        save_ds = save_ds.assign_coords({'latitude': latgrid, 'longitude': longrid, 'category': mcctypes})
        save_ds['category'].attrs['units'] = ds.pred_cat_low.units


        matching_days = np.logical_and(year==yr, dayofyear==day)
        ds_sub_day = ds.isel(num_blocks_low=matching_days)
        
        mcc_grid_save = [] # gotta init this better later on...
        for mcctype in mcctypes:
            matching_type = ds_sub_day.pred_cat_low.values == mcctype
            ds_sub_day_type = ds_sub_day.isel(num_blocks_low=matching_type)
            mccgrid = np.zeros_like(LONS) # to hold counts for this category
            
            igap = .2 # interpolation gap to ensure all boxes are seen by the routine
            ubsave = [] 
            for k in ds_sub_day_type.num_blocks_low.values:
                data = ds_sub_day_type.sel(num_blocks_low=k)
                #so here we get to skip all the estimation of the corners because we saved them!
                # all possible latitude values in a square pattern
                #For Tianle's latlons, bottom left = 2, bottom right = 3, top left = 1, top right = 4
                latsq = np.array([[data.lat1_low,  data.lat4_low], [data.lat2_low, data.lat3_low]])
                latsq_i = interp2d([0,1], [0,1], latsq)(np.arange(0,1,0.2), np.arange(0,1,0.2))
                # all possible longitue values in a square
                lonsq = np.array([[data.lon1_low,  data.lon4_low], [data.lon2_low, data.lon3_low]])
                lonsq_i = interp2d([0,1], [0,1], lonsq)(np.arange(0,1,0.2), np.arange(0,1,0.2))
                    
                # all possible lats/lons pairings within the square
                latloni = list(zip(latsq_i.ravel(), lonsq_i.ravel()))
                # look for unique 1x1 box centers from the vector in latloni                
                uboxes = np.unique(np.floor(latloni)+0.5, axis=0).tolist()
                ubsave.extend(uboxes)

            # boxes at global grid edges
            ubsave = np.array(ubsave)
            ubsave[ubsave<180] = ubsave[ubsave<180]+360;
            ubsave[ubsave>180] = ubsave[ubsave>180]-360;
            
            #add counts for each box:
            for b in ubsave:
                mccgrid[np.logical_and(LATS==b[0],LONS==b[1])] = mccgrid[np.logical_and(LATS==b[0],LONS==b[1])]+1

            mcc_grid_save.append(mccgrid)
        
        #saving netcdf, one per day

        save_ds['counts'] = (('category', 'latitude', 'longitude'), np.array(mcc_grid_save),)

        save_file = os.path.join(target_dir, f'Daily_1x1_MODIS_C6_MCC_{yr}_{day:03.0f}.nc')
        save_ds.to_netcdf(save_file)
        print(save_file)
    #             write out gridded dataset

