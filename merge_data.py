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

verbose=True

# files_sfc_1 = sorted(glob(data+'an_sfc_ERA5_????_max.nc'))
# files_sfc_2 = sorted(glob(data+'an_sfc_ERA5_????_mean.nc'))
# files_sfc_3 = sorted(glob(data+'an_sfc_ERA5_????_min.nc'))
# files_sst = sorted(glob(data+'an_sfc_ERA5_vars2_*mean.nc'))
files_pl = sorted(glob(data+'an_pl_ERA5_*mean.nc'))
# files_land = sorted(glob(data+'an_land_ERA5_*mean.nc'))
# # if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading surface data')
# # sfc_data_1 = xr.open_mfdataset(files_sfc_1, combine="by_coords").sel(lon = slice(-40,100))
# # sfc_data_2 = xr.open_mfdataset(files_sfc_2, combine="by_coords").sel(lon = slice(-40,100))
# # sfc_data_3 = xr.open_mfdataset(files_sfc_3, combine="by_coords").sel(lon = slice(-40,100))
# # if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': computing moisture')
# # sfc_data_2['rh'] = (sfc_data_2.t2m.dims,metpy.calc.relative_humidity_from_dewpoint(sfc_data_2.t2m,sfc_data_2.d2m).data.magnitude)
# # sfc_data_2['q'] = (sfc_data_2.t2m.dims,metpy.calc.mixing_ratio_from_relative_humidity(sfc_data_2.msl, sfc_data_2.t2m, sfc_data_2.rh).data.magnitude)

# # sfc_data_2['tmax'] = sfc_data_1.t2m
# # sfc_data_2['cape'] = sfc_data_1.cape
# # sfc_data_2['cin'] = sfc_data_3.cin

# # if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': writing dataset')
# # sfc_data_2.to_netcdf(scr_data+'EUR_40yr_sfc.nc')

# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading SST data')
# sst_data = xr.open_mfdataset(files_sst, combine="by_coords").sel(lon = slice(-40,100))
# sst_data.to_netcdf(scr_data+'EUR_40yr_sst.nc')

# sfc_data_2['sst'] = sst_data.sst 
# sst_data.close()

# # if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading land data')
# # land_data = xr.open_mfdataset(files_land, combine="by_coords").sel(lon = slice(-40,100))
# # swvl1 = land_data.SWVL1.squeeze()
# # swvl1.to_netcdf(scr_data+'EUR_40yr_land.nc')

if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading PL data')
pl_data = xr.open_mfdataset(files_pl, combine="by_coords").sel(lon = slice(-40,100))

sfc_data_2={}

if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': computing shear')
#sfc_data_2['shear'] = ((pl_data.u.sel(level=500)**2 + pl_data.v.sel(level=500)**2)**0.5 - (pl_data.u.sel(level=900)**2 + pl_data.v.sel(level=900)**2)**0.5).squeeze()
du = pl_data.u.sel(level=500) - pl_data.u.sel(level=900)
dv = pl_data.v.sel(level=500) - pl_data.v.sel(level=900)
sfc_data_2['shear'] = (du**2 + dv**2)**0.5

if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': filling array')
# sfc_data_2['z500'] = pl_data.z.sel(level=500).squeeze()
# sfc_data_2['z900'] = pl_data.z.sel(level=900).squeeze()
sfc_data_2['q900'] = pl_data.q.sel(level=900).squeeze()
sfc_data_2['rh900'] = pl_data.r.sel(level=900).squeeze()
sfc_data_2['q800'] = pl_data.q.sel(level=800).squeeze()
sfc_data_2['rh800'] = pl_data.r.sel(level=800).squeeze()
sfc_data_2['u800'] = pl_data.u.sel(level=800).squeeze()
sfc_data_2['v800'] = pl_data.v.sel(level=800).squeeze()
sfc_data_2['q700'] = pl_data.q.sel(level=700).squeeze()
sfc_data_2['rh700'] = pl_data.r.sel(level=700).squeeze()
sfc_data_2['u700'] = pl_data.u.sel(level=800).squeeze()
sfc_data_2['v700'] = pl_data.v.sel(level=800).squeeze()
sfc_data_2['q600'] = pl_data.q.sel(level=600).squeeze()
sfc_data_2['rh600'] = pl_data.r.sel(level=600).squeeze()
sfc_data_2['u600'] = pl_data.u.sel(level=800).squeeze()
sfc_data_2['v600'] = pl_data.v.sel(level=800).squeeze()
sfc_data_2['q500'] = pl_data.q.sel(level=500).squeeze()
sfc_data_2['rh500'] = pl_data.r.sel(level=500).squeeze()
# sfc_data_2['u500'] = pl_data.u.sel(level=500).squeeze()
# sfc_data_2['v500'] = pl_data.v.sel(level=500).squeeze()
# sfc_data_2['u900'] = pl_data.u.sel(level=900).squeeze()
# sfc_data_2['v900'] = pl_data.v.sel(level=900).squeeze()
# # sfc_data_2['tmax'] = sfc_data_1.t2m
# # sfc_data_2['cape'] = sfc_data_1.cape
# # sfc_data_2['cin'] = sfc_data_3.cin
for key, da in sfc_data_2.items():
    da.name = key
if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': converting array to xarray dataset')
#sfc_data_2 = xr.Dataset(sfc_data_2,compat='override')
sfc_data_2_dataset = xr.merge(list(sfc_data_2.values()), compat='override')
if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': writing dataset')
sfc_data_2_dataset.to_netcdf(scr_data+'EUR_40yr_pl2.nc')

# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading surface data')
# conv={}
# sfc_data_1 = xr.open_mfdataset(files_sfc_1, combine="by_coords").sel(lon = slice(-40,100))
# cape = sfc_data_1.cape
# shear = sfc_data_2['shear']
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': regridding CAPE')
# cape_r = skit.resize(
#         cape.values,
#         shear.values.shape,
#         mode="reflect",  # Handles boundaries gracefully
#         anti_aliasing=True,  # Smooth resizing
#     )
# # regridder = xe.Regridder(cape, shear, method="bilinear")
# # cape_r = regridder(cape)
# conv['conv_EU'] = (cape_r > 500) & (sfc_data_2['shear'] > 10)
# conv['conv_US'] = (cape_r > 1000) & (sfc_data_2['shear'] > 20)

# for key, da in conv.items():
#     da.name = key
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': converting array to xarray dataset')
# #sfc_data_2 = xr.Dataset(sfc_data_2,compat='override')
# conv_dataset = xr.merge(list(conv.values()), compat='override')
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': writing dataset')
# conv_dataset.to_netcdf(scr_data+'EUR_40yr_conv.nc')


# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading surface data')
# conv={}
# sfc_data_1 = xr.open_mfdataset(files_sfc_1, combine="by_coords").sel(lon = slice(-40,100))
# cape = sfc_data_1.cape
# shear = sfc_data_2['shear']
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': regridding CAPE')
# shear_r = skit.resize(
#         shear.values,
#         cape.values.shape,
#         mode="reflect",  # Handles boundaries gracefully
#         anti_aliasing=True,  # Smooth resizing
#     )
# # regridder = xe.Regridder(cape, shear, method="bilinear")
# # cape_r = regridder(cape)
# conv['conv_EU'] = (cape > 500) & (shear_r > 10)
# conv['conv_US'] = (cape > 1000) & (shear_r > 20)

# for key, da in conv.items():
#     da.name = key
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': converting array to xarray dataset')
# #sfc_data_2 = xr.Dataset(sfc_data_2,compat='override')
# conv_dataset = xr.merge(list(conv.values()), compat='override')
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': writing dataset')
# conv_dataset.to_netcdf(scr_data+'EUR_40yr_hrconv.nc')

# #%%
# import argparse as ap

# parser = ap.ArgumentParser(description="Extraction of DI")
# parser.add_argument('-y', '--year', type=int, default=1, help="An integer argument (default: 1)")
# args = parser.parse_args()
# yr = args.year

# path = '/storage/workspaces/giub_meteo_impacts/ci01/CONV/DI/work/meteogroup/DI/era5/grid_gt700/'
# years = np.arange(1980,2023).astype(str)

# year = years[yr]
# files = sorted(glob(path+'/'+year+'/*.cdf'))
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading files of year ', year)
# DI_data = xr.open_mfdataset(files, combine="nested", concat_dim="time", decode_times=False, engine='netcdf4')
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': resetting coordinates')
# start_time = pd.Timestamp(str(year)+"-01-01 00:00")
# correct_time = pd.date_range(start=start_time, periods=DI_data.sizes["time"], freq="3h")
# DI_data = DI_data.assign_coords(time=("time", correct_time))
# DI_data = DI_data.rename({"dimx_N": "lon"})
# DI_data = DI_data.rename({"dimy_N": "lat"})
# new_lon_values = np.linspace(-180, 180, DI_data.sizes["lon"])
# DI_data = DI_data.assign_coords(lon=("lon", new_lon_values))
# new_lat_values = np.linspace(-90, 90, DI_data.sizes["lat"])
# DI_data = DI_data.assign_coords(lat=("lat", new_lat_values))
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': selecting region')
# DI_data = DI_data.sel(lon = slice(-40,100), lat = slice(20,80))
# DI_data = DI_data.sel(time = DI_data.time.dt.month.isin([5,6,7,8,9]))
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': daily aggregation')
# DI_data = DI_data.resample(time="1D").max(skipna=True)

# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': saving data')
# DI_data.to_netcdf(scr_data+year+'EUR_40yr_desea_DI_anom.nc')
# yearly_data = []
# for year in years:
#     if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': opening data ', year)
#     DI_data = xr.open_dataset(scr_data+year+'EUR_40yr_desea_DI_anom.nc')
#     yearly_data.append(DI_data)
# ds_concat = xr.concat(yearly_data, dim="time")
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': saving data')
# ds_concat.to_netcdf(scr_data+'EUR_40yr_desea_DI_anom.nc')

# #%%
# import argparse as ap

# parser = ap.ArgumentParser(description="Extraction of DI")
# parser.add_argument('-y', '--year', type=int, default=1, help="An integer argument (default: 1)")
# args = parser.parse_args()
# yr = args.year

# path = '/storage/workspaces/giub_meteo_impacts/ci01/CONV/wcb_cnn/'
# years = np.arange(1980,2023).astype(str)

# year = years[yr]
# files = sorted(glob(path+'/'+year+'/*/hit*'))
# threshfiles = sorted(glob(path+'thresholds/ocv*'))
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': reading files of year ', year)
# datasets_t = [xr.open_dataset(f) for f in threshfiles]
# datasets_w = [xr.open_dataset(f) for f in files]
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': resetting coordinates')
# start_time = pd.Timestamp(str(year)+"-01-01 00:00")
# correct_time_w = pd.date_range(start=start_time, periods=len(datasets_w), freq="3h")
# correct_time_t = pd.date_range(start=start_time, periods=len(datasets_t), freq="D")
# datasets_t = [datasets_t[f].expand_dims(dim="time").assign_coords(time=("time", [correct_time_t[f]])) for f in range(len(datasets_t))]
# datasets_w = [datasets_w[f].expand_dims(dim="time").assign_coords(time=("time", [correct_time_w[f]])) for f in range(len(datasets_w))]
# WCB_data = xr.concat(datasets_w, dim='time')
# thresh = xr.concat(datasets_t, dim='time')
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': selecting region')
# WCB_data = WCB_data.sel(lon = slice(-40,100), lat = slice(20,80))
# WCB_data = WCB_data.sel(time = WCB_data.time.dt.month.isin([5,6,7,8,9]))
# thresh = thresh.sel(lon = slice(-40,100), lat = slice(20,80))
# thresh = thresh.sel(time = thresh.time.dt.month.isin([5,6,7,8,9]))
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': daily aggregation')
# WCB_data = WCB_data.resample(time="1D").max(skipna=True)
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': thresholding')
# inf = WCB_data.GT800p >= thresh.GT800
# mid = WCB_data.MIDTROPp >= thresh.MIDTROP
# out = WCB_data.LT400p >= thresh.LT400
# all = xr.concat([inf,mid,out],dim='vars').max(dim='vars',skipna=True)
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': saving data')
# inf.to_netcdf(scr_data+year+'EUR_40yr_desea_WCB_inf_anom.nc')
# mid.to_netcdf(scr_data+year+'EUR_40yr_desea_WCB_mid_anom.nc')
# out.to_netcdf(scr_data+year+'EUR_40yr_desea_WCB_out_anom.nc')
# all.to_netcdf(scr_data+year+'EUR_40yr_desea_WCB_all_anom.nc')

# path = '/storage/workspaces/giub_meteo_impacts/ci01/CONV/wcb_cnn/'
# years = np.arange(1980,2022).astype(str)

# yearly_data_inf = []
# yearly_data_mid = []
# yearly_data_out = []
# yearly_data_all = []
# for year in years:
#     if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': opening data ', year)
#     data = xr.open_dataset(scr_data+year+'EUR_40yr_desea_WCB_inf_anom.nc')
#     yearly_data_inf.append(data)
#     data = xr.open_dataset(scr_data+year+'EUR_40yr_desea_WCB_mid_anom.nc')
#     yearly_data_mid.append(data)
#     data = xr.open_dataset(scr_data+year+'EUR_40yr_desea_WCB_out_anom.nc')
#     yearly_data_out.append(data)
#     data = xr.open_dataset(scr_data+year+'EUR_40yr_desea_WCB_all_anom.nc')
#     yearly_data_all.append(data)
# if verbose: print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': saving data')
# ds_concat = xr.concat(yearly_data_inf, dim="time")
# ds_concat.to_netcdf(scr_data+'EUR_40yr_desea_WCB_inf_anom.nc')
# ds_concat = xr.concat(yearly_data_mid, dim="time")
# ds_concat.to_netcdf(scr_data+'EUR_40yr_desea_WCB_mid_anom.nc')
# ds_concat = xr.concat(yearly_data_out, dim="time")
# ds_concat.to_netcdf(scr_data+'EUR_40yr_desea_WCB_out_anom.nc')
# ds_concat = xr.concat(yearly_data_all, dim="time")
# ds_concat.to_netcdf(scr_data+'EUR_40yr_desea_WCB_all_anom.nc')