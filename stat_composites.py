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
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion
import copy
from glob import glob

verbose=True
exp = 'cp100'
region = 10

def generate_random_indices(time_max, num_samples):
    random_time = []
    for _ in range(num_samples):
        time = np.random.randint(5, time_max - 5)
        random_time.append(time)
    return random_time



def get_composite(res,clim,t_p,tag):

    coords1 = {
        'time': np.arange(-5,5,1),  # Time dimension
        'lat': clim.lat,  # Latitude dimension
        'lon': clim.lon,  # Longitude dimension
        'events': np.arange(len(t_p)),  # Event dimension
    }

    fill1 = np.zeros([len(coords1['time']),len(coords1['lat']),len(coords1['lon']),len(t_p)])
    fill1[:] = np.nan

    slices = xr.DataArray(copy.deepcopy(fill1), coords=coords1, dims=['time','lat','lon','events'], name=var+'_std')

    a2=1
    for props in t_p.iterrows():
        if a2%100==0: print(str(100*a2/len(t_p))+' %')
        a2+=1
        prop = props[1]
        time = np.datetime64(prop.time).astype("datetime64[D]")
        itime = np.where(clim.time.astype("datetime64[D]")==time)[0]
        if itime.size>0: itime = itime[0]
        else: print(time, 'no match found', (clim.isel(time=int(prop.itime)).time).astype("datetime64[D]")); continue

        ext = clim.isel(time=slice(itime-5,itime+5)).assign_coords(
            time=coords1['time'],lat=clim.lat,lon=clim.lon)
        
        slices.loc[dict(events=a2 - 2)] = ext

    print('saving slices to netcdf')
    slices.to_netcdf(scr_data+'cookies/static_cookie_'+tag+'.nc')
    print('computing composites')
    composite = slices.mean(dim='events', skipna=True)
    print('saving composites')
    composite.to_netcdf(scr_data+'composite/static_composite_'+tag+'.nc')


def get_composite_randsamp(res,clim,t_p,tag):

    time_max = clim.sizes['time']
    num_random_samples = 100  # Define the number of random samples

    random_coords1 = {
        'time': np.arange(-5,5,1),  # Time dimension
        'lat': clim.lat,  # Latitude dimension
        'lon': clim.lon,  # Longitude dimension
        'events': np.arange(len(t_p)),  # Event dimension
        'samples': np.arange(num_random_samples),
    }
    random_fill1 = np.zeros([len(random_coords1['time']),len(random_coords1['lat']),len(random_coords1['lon']),len(t_p), num_random_samples])
    random_slices = xr.DataArray(copy.deepcopy(random_fill1), coords=random_coords1, dims=['time', 'lat', 'lon', 'events', 'samples'], name=var + '_random_absolute')

    a2=1
    for props in t_p.iterrows():
        if a2%100==0: print(str(100*a2/len(t_p))+' %')
        a2+=1
        prop = props[1]
        time = int(prop.time)

        random_time = generate_random_indices(time_max, num_random_samples)

        # Loop to extract random slices
        for i, (time) in enumerate(random_time):
            random_abs = clim.isel(time=slice(time-5,time+5)).assign_coords(
                time=np.arange(-5, 5, 1),lat=clim.lat,lon=clim.lon)
            
            random_slices.loc[dict(samples=i,events=a2 - 2)] = random_abs

    print('computing composites')
    random_composite = random_slices.mean(dim='events', skipna=True)
    print('saving composites')
    random_composite.to_netcdf(scr_data+'random_composite_'+tag+'.nc')



vars_clim = ['hw','t2m','cp','hwp','msl','cape','cin','rh','SWVL1']
vars_abs_sfc = ['cp']
vars_pl = ['z500','rh900','shear']
vars_spec = ['conv_EU','conv_US']


file=np.load('/storage/homefs/mf23m219/clusters/cluster.npz')
arrays=file.files
labels=file['arr_0']
regs=file['arr_1']
ids=file['arr_2']
coords = {'lat': np.arange(20,60.05,0.25), 'lon': np.arange(-70,30.05,0.25)}
label_N = xr.DataArray((labels==region)*1.0,#(labels==10)*1.0+(labels==3)*1.0+(labels==7)*1.0,
                       dims=['lat','lon'], coords=coords)

t_p = pd.read_csv(scr_data+'cp_conv_obj.csv')


ilat = np.round(t_p.ilat.values).astype(int); ilon = np.round(t_p.ilon.values).astype(int)

rlat = t_p.lat.values
rlon = t_p.lon.values
rlat = np.clip(rlat, np.nanmin(label_N.lat), np.nanmax(label_N.lat))
rlon = np.clip(rlon, np.nanmin(label_N.lon), np.nanmax(label_N.lon))

lat_lon_points = xr.Dataset({"lat": (["points"], rlat), "lon": (["points"], rlon)})

reg = label_N.sel(lat=lat_lon_points["lat"], lon=lat_lon_points["lon"], method="nearest").values
print(np.nansum(reg))
t_p = t_p[reg==1]
t_p = t_p[t_p['size']>100]

t_p.ilat = np.round(t_p.ilat).astype(int)
t_p.ilon = np.round(t_p.ilon).astype(int)

for i1 in range(len(vars_clim))[:]:
    var = vars_clim[i1]
    tag = 'clim'
    file = glob(scr_data+'*desea*'+var+'_*anom.nc')[0]
    ds = xr.open_dataset(file).squeeze().pad(time=(5,5), constant_values=np.nan)
    v = list(ds.data_vars)[0]
    ds = ds[v]
    res=0.25

    get_composite(res,ds,t_p,var+'_'+tag+'_'+exp)
    
    # tag = 'clim_std'
    # ds_std = ds / ds.std(dim='time')

    # get_composite(res,ds_std,t_p,var+'_'+tag+'_'+exp)
    
for i1 in range(len(vars_abs_sfc)):
    var = vars_abs_sfc[i1]; tag = 'abs'
    file = glob(scr_data+'*sfc.nc')[0]
    ds = xr.open_dataset(file)[var].squeeze().pad(time=(5,5), constant_values=np.nan)
    res=0.25

    get_composite(res,ds,t_p,var+'_'+tag+'_'+exp)
        
    # tag = 'abs_std'
    # ds_std = ds / ds.std(dim='time')

    # get_composite(res,ds_std,t_p,var+'_'+tag+'_'+exp)
        
for i1 in range(len(vars_pl)):
    var = vars_pl[i1]; tag = 'clim'
    file = glob(scr_data+'*desea*'+var+'_*anom.nc')[0]
    print(file)
    ds = xr.open_dataset(file).squeeze().pad(time=(5,5), constant_values=np.nan)
    v = list(ds.data_vars)[0]
    ds = ds[v]
    res=0.5
    print(ds.lon)
    get_composite(res,ds,t_p,var+'_'+tag+'_'+exp)
        
    tag = 'clim_std'
    ds_std = ds / ds.std(dim='time')

    get_composite(res,ds_std,t_p,var+'_'+tag+'_'+exp)
        
for i1 in range(len(vars_spec)):
    var = vars_spec[i1]; tag = 'bin'
    file = glob(scr_data+'*hrconv.nc')[0]
    ds = xr.open_dataset(file)[var].squeeze().pad(time=(5,5), constant_values=np.nan)
    res=0.25

    get_composite(res,ds,t_p,var+'_'+tag+'_'+exp)
        
    # tag = 'bin_std'
    # ds_std = ds / ds.std(dim='time')

    # get_composite(res,ds_std,t_p,var+'_'+tag+'_'+exp)