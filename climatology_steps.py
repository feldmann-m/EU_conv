#%% relevant paths
data='/storage/workspaces/giub_meteo_impacts/ci01/CONV/era5/'
figs='/storage/homefs/mf23m219/figs/'
scr_data='/storage/homefs/mf23m219/clim/'
code='/storage/homefs/mf23m219/code/s2s/'
import sys, os
sys.path.append(code)
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
#from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import library.utils as utils
import library.io as io
import skimage.morphology as skimo
import skimage.transform as skit
from scipy.ndimage import convolve
import timeit
from glob import glob
import metpy
from datetime import datetime
import dask
from copy import deepcopy
import dask.array as da
from dask.diagnostics import ProgressBar
import argparse as ap
#import xesmf as xe

parser = ap.ArgumentParser(description="Climatology by variable")
parser.add_argument('-f', '--file', type=int, default=1, help="An integer argument (default: 1)")
parser.add_argument('-n', '--number', type=int, default=1, help="An integer argument (default: 1)")

args = parser.parse_args()

n = args.number
f = args.file

verbose=True

def detrended_roll_climatology(var_da, var, window_size=10, verbose=True):


    rolling_window_size = 10*153

    if verbose: 
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ': Processing '+var)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ': Detrending')

    rolling_mean = var_da.pad(time=(int(rolling_window_size*0.5)), constant_values=np.nan)
    rolling_mean = rolling_mean.rolling(time=rolling_window_size, center=True).mean(skipna=True)
    rolling_mean = rolling_mean.isel(time=slice(int(rolling_window_size*0.5), -int(rolling_window_size*0.5)))
    rolling_mean['time'] = var_da['time']
    detrended_var = var_da - rolling_mean
    if verbose: 
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ': Writing detrended data')
    detrended_var.to_netcdf(scr_data+'EUR_40yr_detr_'+var+'_anom.nc')
    if verbose: 
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ': Step finished')


def deaseasonalized_roll_climatology(var_da, var, window_size=10, verbose=True):

    rolling_window_size = 10

    if verbose: 
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ': Processing '+var)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ': Deseasonalizing')
    
    rolling_mean = var_da.groupby("time.dayofyear").mean("time")

    #rolling_mean = var_da.pad(time=(rolling_window_size // 2), mode='reflect')
    rolling_mean = rolling_mean.rolling(dayofyear=rolling_window_size, center=True).mean()
    deseason_var = (var_da.groupby("time.dayofyear") - rolling_mean).assign_coords(time=var_da.time)
    if verbose: 
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ': Writing deseasonalized data')
    deseason_var.to_netcdf(scr_data+'EUR_40yr_desea_'+var+'_anom.nc')
    if verbose: 
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ': Step finished')




if f==1:
    if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading climate surface data')
    sfc_data_1 = xr.open_mfdataset(scr_data+'EUR_40yr_sfc.nc')
    var_name = [var for var in sfc_data_1.data_vars if 'time' not in var]
    # var_data = sfc_data_1[var_name[n]]
    # del sfc_data_1
    # if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': computing climatology')
    # detrended_roll_climatology(var_data,var_name[n],verbose=verbose)
    # del var_data
    var_data = xr.open_dataset(scr_data+'detr/EUR_40yr_detr_'+var_name[n]+'_anom.nc')
    deaseasonalized_roll_climatology(var_data,var_name[n],verbose=verbose)

if f==2:
    if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading climate pl data')
    sfc_data_2 = xr.open_mfdataset(scr_data+'EUR_40yr_pl2.nc')
    var_name = [var for var in sfc_data_2.data_vars if 'time' not in var]
    var_data = sfc_data_2[var_name[n]]
    # del sfc_data_2
    # if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': computing climatology')
    # detrended_roll_climatology(var_data,var_name[n],verbose=verbose)
    # del var_data
    # var_data = xr.open_dataset(scr_data+'detr/EUR_40yr_detr_'+var_name[n]+'_anom.nc')
    deaseasonalized_roll_climatology(var_data,var_name[n],verbose=verbose)

if f==3:
    if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading climate land data')
    sfc_data_3 = xr.open_mfdataset(scr_data+'EUR_40yr_land.nc')
    var_name = [var for var in sfc_data_3.data_vars if 'time' not in var]
    # var_data = sfc_data_3[var_name[n]]
    # del sfc_data_3
    # if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': computing climatology')
    # detrended_roll_climatology(var_data,var_name[n],verbose=verbose)
    # del var_data
    var_data = xr.open_dataset(scr_data+'detr/EUR_40yr_detr_'+var_name[n]+'_anom.nc')
    deaseasonalized_roll_climatology(var_data,var_name[n],verbose=verbose)

if f==4:
    if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading climate sst data')
    sfc_data_3 = xr.open_mfdataset(scr_data+'EUR_40yr_sst.nc')
    var_name = [var for var in sfc_data_3.data_vars if 'time' not in var]
    var_data = sfc_data_3[var_name[n]]
    # del sfc_data_3
    # if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': computing climatology')
    # detrended_roll_climatology(var_data,var_name[n],verbose=verbose)
    # del var_data
    # var_data = xr.open_dataset(scr_data+'detr/EUR_40yr_detr_'+var_name[n]+'_anom.nc')
    deaseasonalized_roll_climatology(var_data,var_name[n],verbose=verbose)

if f==5:
    if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading synoptic data')
    variable = ['DI','WCB_all','WCB_inf','WCB_mid','WCB_out'][n]
    datavar = ['N','__xarray_dataarray_variable__','__xarray_dataarray_variable__',
               '__xarray_dataarray_variable__','__xarray_dataarray_variable__'][n]
    sfc_data_3 = xr.open_mfdataset(scr_data+'EUR_40yr_'+variable+'.nc')
    if variable=='DI':
        var_data = sfc_data_3.N>0
    else: var_data = sfc_data_3[datavar]
    deaseasonalized_roll_climatology(var_data,variable,verbose=verbose)