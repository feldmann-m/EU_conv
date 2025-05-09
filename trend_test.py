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
# parser.add_argument('-f', '--file', type=int, default=1, help="An integer argument (default: 1)")
parser.add_argument('-n', '--number', type=int, default=1, help="An integer argument (default: 1)")
args = parser.parse_args()
# var = args.file-1
n = args.number

var = ['conv','t2m','cape','rh','z500'][n]

verbose=True

def mk_test(da):
    """Apply Modified Mann-Kendall test to a 1D time series (single grid cell)."""
    if np.all(np.isnan(da)):
        return np.nan  # Return NaN if all values are missing
    result = mk.hamed_rao_modification_test(da,lag=1)
    
    #result = mk.original_test(da)
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

    clim_ym = clim.groupby('time.year').mean('time')
    
    print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': initializing')

    clim_ym = clim_ym.chunk({'lat': 50, 'lon': 50, 'year': -1})
    numeric_time = xr.DataArray(
        np.arange(len(clim_ym.year)),
        dims='year',
        coords={'year': clim_ym.year})
    #((clim_ym['year'] - clim_ym['year'][0]) / np.timedelta64(1, 'Y'))

    fit = clim_ym.polyfit(dim='year', deg=1)
    polyfit_var = [var for var in fit.data_vars if "polyfit_coefficients" in var]
    if not polyfit_var:
        raise ValueError(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),": No polyfit coefficient variable found.")
    polyfit_var = polyfit_var[0]
    slope = fit[polyfit_var].sel(degree=1)
    intercept = fit[polyfit_var].sel(degree=0)

    #trend = intercept + slope * clim.time
    trend = xr.polyval(clim_ym.year, fit[polyfit_var])

    detrended = clim_ym - trend
    std = detrended.std(dim='year')

    slope.to_netcdf(scr_data+'ym_slope_xr_'+tag)
    std.to_netcdf(scr_data+'ym_std_xr_'+tag)


    print(clim_ym.isel(lat=50, lon=50).values)
    print(numeric_time[:5])
    print('mean: ',clim.mean(dim=["lat", "lon"]).values)
    print('std: ',clim.std(dim=["lat", "lon"]).values)


    print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': computing MK-HR for '+tag)

    pval, trend = xr.apply_ufunc(
        mk_test,
        clim_ym,#numeric_time,
        input_core_dims=[['year']],#['year']],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float],
    )
    print(np.isnan(pval).sum().compute())
    print((pval == 0).sum().compute())
    print((trend == 0).sum().compute())
    print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': saving pvalues')
    pval.to_netcdf(scr_data+'ym_pval_hr_'+tag)
    trend.to_netcdf(scr_data+'ym_slope_hr_'+tag)

if verbose: print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading data ', var)
if n==0: var_data = xr.open_mfdataset(scr_data+'EUR_40yr_hrconv.nc').conv_EU
else:
    data = xr.open_mfdataset(scr_data+'EUR_40yr_desea_'+var+'_anom.nc')
    var_name = [var for var in data.data_vars if 'time' not in var]
    var_data = data[var_name[0]]
tag = 'EUR_40yr_'+var+'_trend.nc'
get_trend(var_data, tag)