#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:52:19 2017
@author: jkcm
"""
import pickle
import os
import re
import datetime as dt
import xarray as xr
import numpy as np
import utils
import met_utils as mu
import glob
import netCDF4 as nc4

class CSET_Data:
    """Class for describing generic CSET Data, for saving and loading. To be subclassed only.
    
    Attributes:
        data_location (str): Folder in which this datatype should be saved.
        name (str): unique name of this object, used for saving and loading.
        name_re (regexp obj): Compiled regular expression matching an object's name

    """
    
    def save(self):
        """saves data to disk, specified by data_location and name attributes.
        
        Returns:
            True if successful, otherwise throws exception from pickle.dump
        """
        save_location = os.path.join(self.data_location, self.name + '.pickle')
        print('saving to ' + save_location)
        with open(save_location, 'wb') as f:
            pickle.dump(self, f)
        return True
    
    @classmethod
    def load(cls, data_name):
        """loads specified data file from disk.
    
        Args:
            data_name (str): base name of datafile (not with data_location). Must match name_re
            
        Returns:
            Pickled object stored as data_name
            
        Raises:
            IOError if data_name does not match name_re
            Whatever pickle.load might throw
        """
        # check that input matches data class format and load it
        match = re.match(cls.name_re, data_name.upper())
        if not match:
            raise IOError('cannot recognise data name')
        load_location = os.path.join(cls.data_location, data_name.upper() + '.pickle')
        with open(load_location, 'rb') as f:
            data = pickle.load(f)
            data.check_files()
            return pickle.load(f)
        
class CSET_Flight_Piece(CSET_Data):
    """Class for storing data about an arbitrary piece of a CSET flight.
    Attributes:
        data_location (str): Folder in which this datatype should be saved.
        name (str): unique name of this object, used for saving and loading.
        name_re (regexp obj): Compiled regular expression matching an object's name
        start_time (datetime obj): start date and time of flight piece
        end_time (datetime obj): end date and time of flight piece
    """

    insitu_data_location = r'/home/disk/eos4/jkcm/Data/CSET/flight_data'
    AVAPS_data_location = r'/home/disk/eos4/jkcm/Data/CSET/AVAPS/NETCDF'
    GOES_data_location = r'/home/disk/eos4/jkcm/Data/CSET/GOES/flightpath/GOES_netcdf'
    radarlidar_data_location = r'/home/disk/eos10/imccoy/CSET_RadarLidarData'
    precip_data_location = r'/home/disk/eos4/jkcm/Data/CSET/precip_retrievals'
    chem_data_location = r'/home/disk/eos4/jkcm/Data/CSET/chemistry'
    flightname_re = re.compile('RF(\d\d)')


    def __init__(self, flight_name, start_time, end_time):
        """Initialize flight piece with start/end times, adds in situ data
        flight_name (string): flight name from which this Piece comes, eg RF04
        start_time (datetime obj): flight piece start time
        end_time (datetime obj): flight piece end time
        """
        match = re.match(CSET_Flight_Piece.flightname_re, flight_name.upper())
        if not match:
            raise IOError('cannot recognise flight name')
        self.flight_number = int(match.group(1))
        self.flight_name = flight_name.upper()
        self.start_time = start_time
        self.end_time = end_time
        self.files = dict()#flight_files=[], AVAPS_files=[], GOES_files=[], misc_files=[])
        self.add_insitu_data(self.start_time, self.end_time)
        self.add_GOES_data()

    def check_files(self):
        """Check the existence of all files used to build this Piece,
        raising error if not found"""
        for k, v in self.files:
            print('checking {}...'.format(k))
            for f in v:
                if not os.path.exists(f):
                    raise FileNotFoundError("could not find {} in {}".format(f, k))

    def add_insitu_data(self, start_time, end_time):
        insitu_re = re.compile(self.flight_name + '\.(\d{8})\.(\d{6})_(\d{6})\.PNI\.nc')
        for f in os.listdir(CSET_Flight_Piece.insitu_data_location):
            match = re.match(insitu_re, f)
            if match:
                break
        else:
            raise IOError('no insitu data found for ' + self.name)
        self.insitu_start_date = dt.datetime.strptime(match.group(1) + match.group(2), '%Y%m%d%H%M%S')
        self.insitu_end_date = dt.datetime.strptime(match.group(1) + match.group(3), '%Y%m%d%H%M%S')
        if self.insitu_end_date < self.insitu_start_date:
            self.insitu_end_date += dt.timedelta(days=1)  # if flight passess 0)UTC
        if start_time is None:
            start_time = self.insitu_start_date
        if end_time is None:
            end_time = self.insitu_end_date

        insitu_filename = os.path.join(CSET_Flight.insitu_data_location, match.group(0))
        self.files.setdefault('flight_files', []).append(insitu_filename)
        with xr.open_dataset(insitu_filename) as ds:
            self.flight_data = ds.loc[dict(Time=slice(start_time, end_time))].copy(deep=True)
            self.flight_data.rename({'Time': 'time'}, inplace=True)
        self.add_chemistry()

            
    def add_chemistry(self):
        if not hasattr(self, 'flight_data'):
            raise AttributeError("NO YOU DUMMY, ADD FLIGHT DATA FIRST")
        chem_dict = {"O3": {"long_name": "Fast Ozone mixing ratio", "units": 'ppbv'},
                     "CO": {"long_name": "VUX Carbon Monoxide mixing ratio", "units": 'ppbv'}}
        
        for chem, attrs in chem_dict.items():
            file_glob = os.path.join(CSET_Flight_Piece.chem_data_location, 
                                              "cset-{}_GV_*{}.nc".format(chem, self.flight_name.upper()))
            try:
                chemfile = glob.glob(file_glob)[0]
            except IndexError as e:
                print(file_glob)
                self.flight_data[chem] = (('time'), np.full_like(self.flight_data.GGLAT.values, np.nan))
                self.flight_data[chem] = self.flight_data[chem].assign_attrs(attrs)     
                continue
            dtype = chemfile[-22:-20]
            date = dt.datetime.strptime(chemfile[-16:-8], "%Y%m%d")    
            assert dtype == chem
            try:
                assert date.date() == utils.as_datetime(self.flight_data.time.values[0]).date()
            except AssertionError as e:
                print(date.date())
                print(utils.as_datetime(self.flight_data.time.values[0]).date())
                raise e
            
            with xr.open_dataset(chemfile) as data:
                chem_time = np.array([date + dt.timedelta(seconds=i) for i in data['Start_UTC'].values])
                chem_data = data[dtype]
            data_interp = utils.date_interp(self.flight_data.time.values, chem_time, chem_data, bounds_error=False)
            self.flight_data[chem] = (('time'), data_interp)
            self.flight_data[chem] = self.flight_data[chem].assign_attrs(attrs)       

    def get_variable_by_leg(self, varname, legname, cloud_only=False, flip_cloud_mask=False):
        good_index = self.flight_data['leg'] == legname
        if cloud_only:
            # cloud if ql_cdp > 0.01 g/kg and RH > 95%
            lwc_cdp = self.flight_data['PLWCD_LWOI']
            rhodt = self.flight_data['RHODT']
            mr = self.flight_data['MR']
            cheat_airdens = rhodt/mr
            lwmr_cdp = lwc_cdp/cheat_airdens
            lw_index = lwmr_cdp > 0.01
            RH_index = self.flight_data['RHUM'] > 95
            cloud_index = np.logical_and(RH_index, lw_index)
            if flip_cloud_mask:
                cloud_index = np.logical_not(cloud_index)
            good_index = np.logical_and(good_index, cloud_index)
        vardata = self.flight_data[varname][good_index]
        return vardata
    
    def add_legs(self):
        x = utils.read_CSET_Lookup_Table(rf_num=self.flight_number, legs='all',
                                         sequences='all', variables=['Date', 'ST', 'ET'])
        start_dates = [utils.CSET_date_from_table(x['Date']['values'][i], x['ST']['values'][i])
                       for i in range(len(x['Date']['values']))]
        end_dates = [utils.CSET_date_from_table(x['Date']['values'][i], x['ET']['values'][i])
                     for i in range(len(x['Date']['values']))]
        self.legs = {key: x[key] for key in ['leg', 'rf', 'seq']}
        self.legs['Start'] = np.array(start_dates)
        self.legs['End'] = np.array(end_dates)
#         self.outbound_start_time = min(self.outbound_legs['Start'])
#         self.outbound_end_time = max(self.outbound_legs['End'])
        
        seqs = utils.add_leg_sequence_labels(self.flight_data, 
                                          start_times=self.legs['Start'],
                                          end_times=self.legs['End'],
                                          legs=self.legs['leg'],
                                          sequences=self.legs['seq'])
        self.sequences = sorted(list(set(seqs)))


    
    def get_profiles(self, seqs=None):
        if not 'leg' in self.flight_data.coords.keys():
                print("leg/sequence labels not added")
                return None
        
        profiles = {}
        if not seqs:
            seqs = self.sequences
        for seq in seqs:
            d = self.flight_data.where(
                np.logical_and(self.flight_data.leg=='d', self.flight_data.sequence==seq), drop=True)
            var_list = ['GGLAT', 'GGLON', 'GGALT', 'RHUM', 'ATX', 'MR', 'THETAE', 'THETA', 'PSXC', 'DPXC', 'PLWCC']    
            

            sounding_dict = {}
            sounding_dict['TIME'] = d.time.values
            for i in var_list:
                sounding_dict[i] = d[i].values
            if 'ATX' in var_list:
                sounding_dict['ATX'] = sounding_dict['ATX'] + 273.15
                

            sounding_dict['DENS'] = mu.density_from_p_Tv(d['PSXC'].values*100, d['TVIR'].values+273.15)  
            sounding_dict['QL'] = d['PLWCC'].values/sounding_dict['DENS']
            sounding_dict['THETAL'] = mu.get_liquid_water_theta(
                sounding_dict['ATX'], sounding_dict['THETA'], sounding_dict['QL'])
            sounding_dict['QV'] = d['MR'].values/(1+d['MR'].values/1000)

            decoupling_dict = mu.calc_decoupling_from_sounding(sounding_dict, usetheta=False)
            zi_dict = mu.calc_zi_from_sounding(sounding_dict)
            profiles[seq] = {"data": d, "dec": decoupling_dict, "zi": zi_dict, "sounding": sounding_dict}
        return profiles
        
    def add_ERA_data(self):
        """Retrieve ERA5 data in a box around a trajectory
        Assumes ERA5 data is 0.3x0.3 degrees
        Returns an xarray Dataset
        """
        start = utils.as_datetime(self.flight_data.time.values[0]).replace(minute=0, second=0)
        end = utils.as_datetime(self.flight_data.time.values[-1]).replace(minute=0, second=0)+dt.timedelta(hours=1)
        dates = np.array([start + dt.timedelta(minutes=i*15) for i in range(1+int((end-start).total_seconds()/(60*15)))])
        index = [np.argmin(abs(utils.as_datetime(self.flight_data.time.values) - i)) for i in dates]
        lats = self.flight_data.GGLAT.values[index]
        lons = self.flight_data.GGLON.values[index]
        times = [np.datetime64(i.replace(tzinfo=None)) for i in dates]
        box_degrees = 2
        space_index = int(np.round(box_degrees/0.3/2)) # go up/down/left/right this many pixels
        unique_days = set([utils.as_datetime(i).date() for i in times])
        files = [os.path.join(utils.ERA_source, "ERA5.pres.NEP.{:%Y-%m-%d}.nc".format(i))
                 for i in unique_days]
        sfc_files = [os.path.join(utils.ERA_source, "ERA5.sfc.NEP.{:%Y-%m-%d}.nc".format(i))
                 for i in unique_days]
        flux_files = [os.path.join(utils.ERA_source, "4dvar_sfc_proc", "ERA5.4Dvarflux.NEP.{:%Y-%m-%d}.nc".format(i))
                 for i in unique_days]
        self.files['ERA_files'] = files + sfc_files
        with xr.open_mfdataset(sorted(files)) as data:
            #return_ds = xr.Dataset(coords={'time': ds.coords['time'], 'level': data.coords['level']})
            ds = xr.Dataset(coords={'time': (('time'), times, data.coords['time'].attrs),
                                    'level': (('level'), data.coords['level'])})

 
            # ds.coords['level'] = data.coords['level']

            #adding in q:
            T = data['t'].values 
            RH = data['r'].values
            p = np.broadcast_to(data.coords['level'].values[None, :, None, None], T.shape)*100
            q = utils.qv_from_p_T_RH(p, T, RH)
            data['q'] = (('time', 'level', 'latitude', 'longitude'), q)
            data['q'] = data['q'].assign_attrs({'units': "kg kg**-1", 
                                    'long_name': "specific_humidity",
                                    'dependencies': 'ERA_t, ERA_p, ERA_r'})

            # adding gradients in for z, t, and q. Assuming constant grid spacing.
            for var in ['t', 'q', 'z', 'u', 'v']:
                [_,_,dvardj, dvardi] = np.gradient(data[var].values)
                dlatdy = 360/4.000786e7  # degrees lat per meter y
                def get_dlondx(lat) : return(360/(np.cos(np.deg2rad(lat))*4.0075017e7))

                lat_spaces = np.diff(data.coords['latitude'].values)
                lon_spaces = np.diff(data.coords['longitude'].values)
                assert(np.allclose(lat_spaces, -0.3, atol=0.01) and np.allclose(lon_spaces, 0.3, atol=0.05))
                dlondi = np.mean(lon_spaces)
                dlatdj = np.mean(lat_spaces)
                dlondx = get_dlondx(data.coords['latitude'].values)
                dvardx = dvardi/dlondi*dlondx[None,None,:,None]
                dvardy = dvardj/dlatdj*dlatdy
                data['d{}dx'.format(var)] = (('time', 'level', 'latitude', 'longitude'), dvardx)
                data['d{}dy'.format(var)] = (('time', 'level', 'latitude', 'longitude'), dvardy)

            grad_attrs = {'q': {'units': "kg kg**-1 m**-1",
                                'long_name': "{}_gradient_of_specific_humidity",
                                'dependencies': "ERA_t, ERA_p, ERA_r"},
                          't':  {'units': "K m**-1",
                                'long_name': "{}_gradient_of_temperature",
                                'dependencies': "ERA_t"},
                          'z':  {'units': "m**2 s**-2 m**-1",
                                'long_name': "{}_gradient_of_geopotential",
                                'dependencies': "ERA_z"},
                          'u': {'units': "m s**-1 m**-1",
                                'long_name': "{}_gradient_of_zonal_wind",
                                'dependencies': "ERA_u"},
                          'v': {'units': "m s**-1 m**-1",
                                'long_name': "{}_gradient_of_meridional_wind",
                                'dependencies': "ERA_v"}}

            for key, val in grad_attrs.items():
                for (n, drn) in [('x', 'eastward'), ('y', 'northward')]:
                    attrs = val.copy()
                    var = 'd{}d{}'.format(key, n)
                    attrs['long_name'] = attrs['long_name'].format(drn)
                    data[var] = data[var].assign_attrs(attrs)

            for var in data.data_vars.keys():
                vals = []
                for (lat, lon, time) in zip(lats, lons%360, times):
                    if lat > np.max(data.coords['latitude']) or lat < np.min(data.coords['latitude']) or \
                        lon > np.max(data.coords['longitude']) or lon < np.min(data.coords['longitude']):
                        print('out of range of data')
                        print(lat, lon, time)
                        vals.append(np.full_like(data.coords['level'], float('nan'), dtype='float'))
                        continue
                    x = data[var].sel(longitude=slice(lon - box_degrees/2, lon + box_degrees/2),
                                      latitude=slice(lat + box_degrees/2, lat - box_degrees/2))
                    z = x.sel(method='nearest', time=time, tolerance=np.timedelta64(1, 'h'))
                    #z = y.sel(method='nearest', tolerance=50, level=pres)
                    #this applies a 2D gaussian the width of z, i.e. sigma=box_degrees
                    # print(z.shape)
                    gauss = utils.gauss2D(shape=z.shape[1:], sigma=z.shape[0])
                    filtered = z.values * gauss
                    vals.append(np.sum(filtered, axis=(1,2)))
                ds['ERA_'+var] = (('time', 'level'), np.array(vals))
                ds['ERA_'+var] = ds['ERA_'+var].assign_attrs(data[var].attrs)


            t_1000 = ds.ERA_t.sel(level=1000).values
            theta_700 = mu.theta_from_p_T(p=700, T=ds.ERA_t.sel(level=700).values)
            LTS = theta_700-t_1000
            ds['ERA_LTS'] = (('time'), np.array(LTS))
            ds['ERA_LTS'] = ds['ERA_LTS'].assign_attrs(
                    {"long_name": "Lower tropospheric stability",
                     "units": "K",
                     "_FillValue": "NaN"})
            t_dew = t_1000-(100-ds.ERA_r.sel(level=1000).values)/5
            lcl = mu.get_LCL(t=t_1000, t_dew=t_dew, z=ds.ERA_z.sel(level=1000).values/9.81)
            z_700 = ds.ERA_z.sel(level=700).values/9.81
            gamma_850 = mu.get_moist_adiabatic_lapse_rate(ds.ERA_t.sel(level=850).values, 850)
            eis = LTS - gamma_850*(z_700-lcl)
            ds['ERA_EIS'] = (('time'), np.array(eis))
            ds['ERA_EIS'] = ds['ERA_EIS'].assign_attrs(
                    {"long_name": "Estimated inversion strength",
                     "units": "K",
                     "_FillValue": "NaN"})
            
            
            with xr.open_mfdataset(sorted(sfc_files)) as sfc_data:
                for var in sfc_data.data_vars.keys():
                    vals = []
                    for (lat, lon, time) in zip(lats, lons%360, times):
                        if lat > np.max(sfc_data.coords['latitude']) or lat < np.min(sfc_data.coords['latitude']) or \
                            lon > np.max(sfc_data.coords['longitude']) or lon < np.min(sfc_data.coords['longitude']):
                            print('out of range of data')
                            print(lat, lon, time)
                            vals.append(float('nan'))
                            continue
                        x = sfc_data[var].sel(longitude=slice(lon - box_degrees/2, lon + box_degrees/2),
                                              latitude=slice(lat + box_degrees/2, lat - box_degrees/2))
                        z = x.sel(method='nearest', time=time, tolerance=np.timedelta64(1, 'h'))
                        gauss = utils.gauss2D(shape=z.shape, sigma=z.shape[0])
                        filtered = z.values * gauss
                        vals.append(np.sum(filtered))
                    ds['ERA_'+var] = (('time'), np.array(vals))
                    ds['ERA_'+var] = ds['ERA_'+var].assign_attrs(sfc_data[var].attrs)
                    
            with xr.open_mfdataset(sorted(flux_files)) as flux_data:
                for var in flux_data.data_vars.keys():
                    if var not in ['sshf', 'slhf']:
                        continue
                    vals = []
                    for (lat, lon, time) in zip(lats, lons%360, times):
                        if lat > np.max(flux_data.coords['latitude']) or lat < np.min(flux_data.coords['latitude']) or \
                            lon > np.max(flux_data.coords['longitude']) or lon < np.min(flux_data.coords['longitude']):
                            print('out of range of data')
                            print(lat, lon, time)
                            vals.append(float('nan'))
                            continue
                        x = flux_data[var].sel(longitude=slice(lon - box_degrees/2, lon + box_degrees/2),
                                              latitude=slice(lat + box_degrees/2, lat - box_degrees/2))
                        z = x.sel(method='nearest', time=time, tolerance=np.timedelta64(1, 'h'))
                        gauss = utils.gauss2D(shape=z.shape, sigma=z.shape[0])
                        filtered = z.values * gauss
                        vals.append(np.sum(filtered))
                    ds['ERA_'+var] = (('time'), np.array(vals))
                    ds['ERA_'+var] = ds['ERA_'+var].assign_attrs(flux_data[var].attrs)

                
        self.ERA_data = ds
        
    def add_GOES_data(self):
        self.GOES_data = {}
        for res in ['1deg', '2deg', '4deg']:
            goes_file = os.path.join(CSET_Flight_Piece.GOES_data_location, '-'.join([self.flight_name, res + '.nc']))
            with xr.open_dataset(goes_file) as data:
                self.GOES_data[res] = data.sel(time=slice(self.start_time, self.end_time)) 
                
    def add_radarlidar_data(self):
        radarlidar_file = os.path.join(CSET_Flight_Piece.radarlidar_data_location,
                                    '{}_COMBINED_HCR_HSRL_data_mask_version4.cdf'.format(self.flight_name.upper()))
        with xr.open_dataset(radarlidar_file) as data:
            data.rename({'absolute_time': 'time'}, inplace=True)
            data['time'].values = data.time.values.astype('datetime64')
            self.radarlidar_data = data.sel(time=slice(self.start_time, self.end_time)) 
            # self.radarlidar_data =  data.where(np.logical_and(data.time>self.start_time, data.time<self.end_time))
            
    def add_precip_data(self):
        precip_files = glob.glob(os.path.join(CSET_Flight_Piece.precip_data_location,
                                  '{}_*.cdf'.format(self.flight_name.upper())))
        if not len(precip_files) == 1:
            raise IOError("incorrect number of precip files found")
        with xr.open_dataset(precip_files[0]) as data:
            data['time'].values = data.time.values.astype('datetime64')
            self.precip_data = data.sel(time=slice(self.start_time, self.end_time)) 

    def get_max_precip_by_leg(self, legname):
        if not hasattr(self, 'precip_data'):
            raise ValueError("You haven't added the precip data yet, buddy")
        good_index = self.precip_data['leg'] == legname
        prec = self.precip_data['lwf'][good_index]
        return prec.max(dim='height', skipna=True)   
    
    



class CSET_Flight_Sequence(CSET_Flight_Piece):

    def __init__(self, start_time, end_time):
        super().__init__(start_time, end_time)
        self.type="flight_sequence"


class CSET_Flight(CSET_Flight_Piece):
    data_location = r'/home/disk/eos4/jkcm/Data/CSET/Python/flights'
    name_re = re.compile('RF(\d\d)')

    
    def __init__(self, flight_name):
        match = re.match(CSET_Flight.name_re, flight_name.upper())
        if not match:
            raise IOError('cannot recognise flight name')
        self.flight_number = int(match.group(1))
        self.name = flight_name.upper()
        self.flight_name = flight_name.upper()
        self.files = dict()
        self.add_insitu_data(start_time=None, end_time=None)
        self.add_GOES_data()
        
        self.start_time = utils.as_datetime(self.flight_data.time.values[0])
        self.end_time = utils.as_datetime(self.flight_data.time.values[-1])
    
        

    def add_AVAPS_data(self):
        pass
    
    def add_GOES_data(self):
        self.GOES_data = {}
        for res in ['1deg', '2deg', '4deg']:
            filename = os.path.join(CSET_Flight.GOES_data_location, '-'.join([self.name, res + '.nc']))
            if not os.path.exists(filename):
                raise IOError('could not find GOES data')
            self.GOES_data[res] = xr.open_dataset(filename)

    def add_ERA5_data(self):
        pass

class dep_Flight:
    """A research flight for CSET
    
    Attributes:
        name: 
        direction:
        takeoff_date
        
    Methods:
        
    """
    
    def __repr__(self):
        return(self.name+":\n"+"\n".join(self.__dict__.keys()))
    
    
    def __init__(self, flight_name):
        parsed_name = re.match(r'rf(\d\d)', flight_name)
        if not parsed_name:
            raise ValueError('flight_name not of form \'rf##\', please enter valid flight name')
        self.name = flight_name
        self.flight_number = int(parsed_name.group(1))
        self.direction = 'outbound' if self.flight_number % 2 == 0 else 'return'
        self.takeoff_date, self.landing_date = self.lookup_flight_start_end_times()
        self.pair_flight_number = self.flight_number + 1 if self.direction == 'outbound' \
            else self.flight_number - 1 
        self.flight_pair = 'rf{:02d}'.format(self.pair_flight_number)
        
        # Setting all associated file locations
        g = glob.glob(os.path.join(
                params.flight_data_dir, '{}*.nc'.format(self.name.upper())))
        if len(g) is not 1:
            raise IOError('could not identify aircraft data file')
        else: self.aircraft_data_file = g[0]
        
        g = glob.glob(os.path.join(
                params.hirate_data_dir, '{}*.nc'.format(self.name.upper())))
        if len(g) is not 1:
            raise IOError('could not identify aircraft hirate data file')
        else: self.hirate_aircraft_data_file = g[0]

        g = glob.glob(os.path.join(
                params.a_waypts, '*{}*.txt'.format(self.name.upper())))
        if len(g) is not 1:
            raise IOError('could not identify a-waypoints data file')
        else: self.a_waypoints_file = g[0]

        g = glob.glob(os.path.join(
                params.b_waypts, '*{}*.txt'.format(self.name.upper())))
        if len(g) is not 1:
            raise IOError('could not identify b-waypoints data file')
        else: self.b_waypoints_file = g[0]
        
        g= glob.glob(os.path.join(
                params.sausage_dir, '*{}*'.format(self.name.lower())))
        if len(g) is not 1:
            raise IOError('could not identify sausage data file')
        else: self.sausage_file = g[0]

        self.AVAPS_data_filelist = self.get_AVAPS_data_filelist()
    
        self.flightpath_goes_data_filedict = self.lookup_flightpath_goes_data_filedict()

        self.trajectory_goes_data_filedict = self.lookup_trajectory_goes_data_filedict()
        
        self.trajectories_filedict = self.lookup_trajectories_filedict() 
        

    
    def lookup_flight_start_end_times(self):
        x = read_CSET_Lookup_Table(params.CSET_lookuptable, 
                                   rf_num=self.flight_number,
                                   sequences=['m', 'k'], 
                                   variables=['Date', 'ST', 'ET'])
        # deal with ordering of mather and kona legs between flight directions
#        (s_indx, e_indx) = (0, 1) if self.direction == 'outbound' else (0, 1)
#        start_time = CSET_date_from_table(
#                x['Date']['values'][s_indx], x['ST']['values'][s_indx])
#        end_time = CSET_date_from_table(
#                x['Date']['values'][e_indx], x['ET']['values'][e_indx])
        start_time = CSET_date_from_table(
                x['Date']['values'][0], x['ST']['values'][0])
        end_time = CSET_date_from_table(
                x['Date']['values'][1], x['ET']['values'][1])
        return start_time, end_time

    def lookup_flightpath_goes_data_filedict(self):
        filedict = {}
        for res in ['1deg', '2deg', '4deg']:
            g = glob.glob(os.path.join(
                params.goes_flightpath_dir, '{}-{}.nc'.format(self.name.upper(), res)))
            if len(g) is not 1:
                raise IOError('could not identify flightpath_goes data file')
            else: 
                filedict[res] = g[0]
        return filedict
        #nested dict by resolution, then by trajectory name
        pass
    
    def lookup_trajectory_goes_data_filedict(self):
        filedict = {}
        for res in ['1deg', '2deg', '4deg']:
            g = glob.glob(os.path.join(
                params.goes_trajectory_dir, '*{}*{}.nc'.format(self.name.lower(), res)))
            if len(g) < 1:
                raise IOError('could not identify trajectory_goes data file')
            else: 
                filedict[res] = g
        return filedict
            
    def get_AVAPS_data_filelist(self):
        avaps_filelist = []
        g = glob.glob(os.path.join(params.AVAPS_dir, '*.nc'))
        for avaps_file in g:
            avaps_date = dt.datetime.strptime(os.path.basename(avaps_file), 
                                              'D%Y%m%d_%H%M%S_PQC.nc')
            if self.takeoff_date < avaps_date < self.landing_date:
                avaps_filelist.append(avaps_file)
        return avaps_filelist
    
    

    def lookup_trajectories_filedict(self):
        filedict = {}
        g = os.path.join(params.trajects, '{}_{}'.format(*sorted([self.name, self.flight_pair])))
        if not os.path.exists(g):
            raise IOError('could not identify trajectory directory')
        for traj_type in ['1000m_+72', '1000m_-48', '500m_+72', '500m_-48']:
            files = glob.glob(os.path.join(g, 'analysis*{}.txt'.format(traj_type)))
            filedict[traj_type] = files
        return filedict
    
    
    def save(self, filename):
        pass
    def load(filename):
        #load the file, touch all the files to make sure they exist
        pass
