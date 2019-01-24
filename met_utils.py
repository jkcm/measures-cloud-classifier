# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:42:46 2018

@author: jkcm
"""
import numpy as np

######################################
### CONSTANTS ###
######################################
p0 = 1000.  # reference pressure, hPa
Rdry = 287.  # gas const for dry air, J/K/kg
Rvap = 461.  # gas const for water vapor, J/K/kg
eps = Rvap/Rdry - 1
cp = 1004.  # cp_dry, specific heat of dry air at const pressure, J/K/kg
g = 9.81   # grav acceleration at sea level, m/s2
lv = 2.5*10**6  # latent heat of vaporization at 0C, J/kg


def theta_from_p_t(p, t, p0=1000):
    """calculate potential temperature from pressure, temperature
    p: pressure in hPa
    t: temperature in Kelvin
    
    returns: potential temperature in Kelvin
    """
    theta = t * (p0/p)**(Rdry/cp)
    return theta

def calculate_moist_adiabatic_lapse_rate(t, p): 
    """calculate moist adiabatic lapse rate from pressure, temperature
    p: pressure in hPa
    t: temperature in Kelvin
    
    returns: moist adiabatic lapse rate in Kelvin/m
    """
    es = 611.2*np.exp(17.67*(t-273.15)/(t-29.65)) # Bolton formula, es in Pa
    qs = 0.622*es/(p*100-0.378*es)
    num = 1 + lv*qs/(Rdry*t)
    denom = 1 + lv**2*qs/(cp*Rvap*t**2)
    gamma = g/cp*(1-num/denom)
    return gamma

def calculate_LTS(t_700, t_1000):
    """calculate lower tropospheric stability
    t_700: 700 hPa temperature in Kelvin
    t_1000: 1000 hPa temperature in Kelvin
    
    returns: lower tropospheric stability in Kelvin
    """
    theta_700 = theta_from_p_t(p=700, t=t_700)
    lts = theta_700-t_1000
    return lts
    
def calculate_LCL(t, t_dew, z=0):
    """calculate lifting condensation level from temperature, dew point, and altitude
    t: temperature in Kelvin
    t_dew: dew point temperature in Kelvin
    z: geopotential height in meters. defaults to 0
    
    returns: lifting condensation level in meters
    
    raises: ValueError if any dew points are above temperatures (supersaturation)
    """
    if np.any(t_dew > t):
        raise ValueError('dew point temp above temp, that\'s bananas')
    return z + 125*(t - t_dew)
    
def calculate_EIS(t_1000, t_850, t_700, z_1000, z_700, r_1000):
    """calculate estimated inversion strength from temperatures, heights, relative humidities
    t_1000, t_850, t_700: temperature in Kelvin at 1000, 850, and 700 hPa
    z_1000, z_700: geopotential height in meters at 1000 and 700 hPa
    r_1000: relative humidity in % at 1000 hPa
    
    returns: estimated inversion strength (EIS) in Kelvin
    """
    if hasattr(r_1000, '__iter__'):
        r_1000[r_1000>100] = 100  # ignoring supersaturation for lcl calculation
    t_dew = t_1000-(100-r_1000)/5
    lcl = calculate_LCL(t=t_1000, t_dew=t_dew, z=z_1000)
    lts = calculate_LTS(t_700=t_700, t_1000=t_1000)
    gamma_850 = calculate_moist_adiabatic_lapse_rate(t=t_850, p=850)
    eis = lts - gamma_850*(z_700-lcl)
    return eis

def calc_EIS_from_ERA5(ERA_data, add_to_dataset=False):
    """convenience function for calculating EIS from ERA5 datasets
    ERA_data: xarray-like dataset containing pressure level ERA5 data with standard names
    add_to_dataset: boolean; whether to update the dataset with a new variable or not.
    
    returns: EIS in Kelvin
    
    raises: IOError if ERA_data is not in expected format
    """
    if not all([i in ERA_data.data_vars for i in ['t', 'z', 'r']]):
        raise IOError('ERA_data does not have necessary vars: t, z, r')
    if not 'level' in ERA_data.coords:
        raise IOError('ERA_data not on pressure levels')
    if not all([i in ERA_data.level.values for i in [700, 850, 1000]]):
        raise IOError('ERA_data does not have necessary pressure levels: 700, 850, 100')
    t_1000 = ERA_data.t.sel(level=1000).values
    t_850 = ERA_data.t.sel(level=850).values
    t_700 = ERA_data.t.sel(level=700).values
    z_1000 = ERA_data.z.sel(level=1000).values/g
    z_700 = ERA_data.z.sel(level=700).values/g
    r_1000 = ERA_data.r.sel(level=1000).values

    eis= calculate_EIS(t_1000=t_1000, t_850=t_850, t_700=t_700, 
                       z_1000=z_1000, z_700=z_700,
                       r_1000=r_1000)
    if add_to_dataset:
        ds['EIS'] = (('time'), np.array(eis), 
                     {"long_name": "Estimated inversion strength",
                      "units": "K",
                      "_FillValue": "NaN"})
    return eis

def density_from_p_Tv(p, Tv):
    return p/(Rdry*Tv)


def get_liquid_water_theta(temp, theta, q_l):
    """temp = air temp (K) theta = pot temp, q_l = liquid water MR"""
    theta_l = theta - (theta*lv*q_l/(temp*cp*1000))
    return theta_l




if __name__ == "__main__":
    import xarray as xr
    ERA_testfile = r'/home/disk/eos4/jkcm/Data/CSET/ERA5/ERA5.pres.NEP.2015-08-25.nc'
    ERA_data = xr.open_dataset(f)
    EIS = calc_EIS_from_ERA5(ERA_data)
    