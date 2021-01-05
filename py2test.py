# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:53:54 2019

@author: Hans
"""

#
#t, p0, p, Rdry, cp = 268, 1000, 700, 287, 1004.0
#rat = Rdry/cp
#rat2 = p0/p
#print(rat)
#print(rat2)
#theta_700 = t * (p0/p)**rat
#print(theta_700)



from met_utils import *
t_1000, t_850, t_700, z_1000, z_700, r_1000 = 288.0,282.0,268.0,100.0,3400.0,90.0
print(calculate_EIS(t_1000, t_850, t_700, z_1000, z_700, r_1000))

#if hasattr(r_1000, '__iter__'):
#    r_1000[r_1000>100] = 100  # ignoring supersaturation for lcl calculation
#t_dew = t_1000-(100-r_1000)/5
#lcl = calculate_LCL(t=t_1000, t_dew=t_dew, z=z_1000)
#print(lcl)
##lts = calculate_LTS(t_700=t_700, t_1000=t_1000)
##theta_700 = theta_from_p_t(p=700, t=t_700)
#p, t, p0 = 700, t_700, 1000
#print(t)
#print(p0)
#print(p)
#print(Rdry)
#print(cp)
#theta_700 = t * (p0/p)**(Rdry/cp)
#
#print(theta_700)
#lts = theta_700-t_1000
#print(lts)
#gamma_850 = calculate_moist_adiabatic_lapse_rate(t=t_850, p=850)
#print(gamma_850)
#eis = lts - gamma_850*(z_700-lcl)
#print(eis)