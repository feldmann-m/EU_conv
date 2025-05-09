#%% relevant paths
data='/storage/workspaces/giub_meteo_impacts/ci01/CONV/era5/'
figs='/storage/homefs/mf23m219/figs/'
scr_data='/storage/homefs/mf23m219/clim/'#detr_desea/'
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
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion
from sklearn.decomposition import PCA
import copy
from glob import glob
import argparse as ap
import scipy.stats as stats
import pymannkendall as mk
from scipy.stats import linregress
#import xesmf as xe

parser = ap.ArgumentParser(description="Climatology by variable")
parser.add_argument('-f', '--file', type=int, default=1, help="An integer argument (default: 1)")
parser.add_argument('-n', '--number', type=int, default=1, help="An integer argument (default: 1)")
args = parser.parse_args()
dset = args.file-1
n = args.number

dsets = ['EUR_40yr_hrconv.nc','EUR_40yr_land.nc','EUR_40yr_pl.nc','EUR_40yr_sfc.nc','EUR_40yr_sst.nc']

verbose=True

def mk_test(da):
    """Apply Modified Mann-Kendall test to a 1D time series (single grid cell)."""
    if np.all(np.isnan(da)):
        return np.nan  # Return NaN if all values are missing
    #result = mk.hamed_rao_modification_test(da,lag=1)
    
    result = mk.original_test(da)
    return result.p, result.slope

def linear_regression(x,time):
    x = np.asarray(x)
    time = np.asarray(time)
    mask = ~np.isnan(time) & ~np.isnan(x)
    if np.sum(mask) < 2:  # Need at least two points to compute a regression
        return np.nan, np.nan

    x_clean = x[mask]
    time_clean = time[mask]
    slope, intercept, r_value, p_value, std_err = linregress(time_clean,x_clean)
    return p_value, slope

def get_trend(clim,tag):

    clim = clim.sortby('time',ascending=True)
    clim["time"] = clim["time"].dt.floor("D")
    print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': initializing')

    clim = clim.chunk({'lat': 50, 'lon': 50, 'time': -1})
    numeric_time = ((clim['time'] - clim['time'][0]) / np.timedelta64(1, 'D'))
    print(clim.isel(lat=50, lon=50).values)
    print(numeric_time.values[:5])
    print('mean: ',clim.mean(dim=["lat", "lon"]).values)
    print('std: ',clim.std(dim=["lat", "lon"]).values)


    print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': computing MK-HR for '+tag)

    trend, pval = xr.apply_ufunc(
        linear_regression,
        clim,numeric_time,
        input_core_dims=[['time'],['time']],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float],
    )
    print(np.isnan(pval).sum().compute())
    print((pval == 0).sum().compute())
    print((trend == 0).sum().compute())
    print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': saving pvalues')
    pval.to_netcdf(scr_data+'pval_lr_'+tag)
    trend.to_netcdf(scr_data+'slope_lr_'+tag)

if verbose: print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading data ',dsets[dset])
data = xr.open_mfdataset(scr_data+dsets[dset])
var_name = [var for var in data.data_vars if 'time' not in var]
var_data = data[var_name[n]]
tag = 'EUR_40yr_slope_'+var_name[n]+'.nc'
get_trend(var_data, tag)