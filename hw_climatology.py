#%% relevant paths
data='/storage/workspaces/giub_meteo_impacts/ci01/CONV/era5/'
figs='/storage/homefs/mf23m219/figs/'
scr_data='/storage/homefs/mf23m219/clim/detr_desea/'
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
import timeit
from glob import glob
import metpy
from datetime import datetime
import dask

verbose=True

if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading temperature data')
var='tmax'
des_t2m = xr.open_mfdataset(scr_data+'EUR_40yr_desea_'+var+'_anom.nc')

if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': computing heatwaves')
#des_t2m = deseasonalized_xr.t2m#xr.open_dataarray(scr_data+'t2m_xr.nc', chunks={"time": 100})

threshold_heatwave = des_t2m.chunk({"time": -1}).quantile(0.95,dim=["time"])
heatwaves = des_t2m >= threshold_heatwave
if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': writing heatwaves')
heatwaves.to_netcdf(scr_data+'EUR_40yr_desea_hw_anom.nc')
if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': computing persistent heatwaves')
heatwaves = xr.open_mfdataset(scr_data+'EUR_40yr_desea_hw_anom.nc')
selem = np.ones((3, 1, 1))
hw = heatwaves.tmax.values
closed_hw = skimo.binary_closing(hw, selem)
heatwaves['tmax'] = (heatwaves.tmax.dims, closed_hw)
if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': writing persistent heatwaves')
heatwaves.to_netcdf(scr_data+'EUR_40yr_desea_hwp_anom.nc')
