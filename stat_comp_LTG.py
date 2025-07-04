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
# import library.utils as utils
# import library.io as io
import skimage.morphology as skimo
import skimage.transform as skit
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion
from sklearn.decomposition import PCA
import copy
from glob import glob
import argparse as ap
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import pymannkendall as mk
from scipy.stats import linregress
import warnings
#import xesmf as xe

parser = ap.ArgumentParser(description="Climatology by variable")
parser.add_argument('-r', '--region', type=int, default=1, help="An integer argument (default: 1)")
# parser.add_argument('-v', '--variable', type=int, default=1, help="An integer argument (default: 1)")
# parser.add_argument('-t', '--type', type=int, default=1, help="An integer argument (default: 1)")
args = parser.parse_args()

reg = args.region
# v0 = args.variable
# t0 = args.type

verbose=True
exp = 'conv100_test'

#conv_EU = xr.open_dataset(scr_data+'EUR_40yr_hrconv.nc').conv_EU
cape = xr.open_dataset(scr_data+'EUR_40yr_sfc.nc').cape
print(cape.values.shape,cape.time.dt.year)
cape = cape.sel(time=cape.time.dt.year.isin(np.arange(2001,2024)))
shear = xr.open_dataset(scr_data+'EUR_40yr_pl.nc').shear
shear = shear.sel(time=shear.time.dt.year.isin(np.arange(2001,2024)))
print(cape.values.shape,shear.values.shape)
shear_r = skit.resize(
        shear.values,
        cape.values.shape,
        mode="reflect",  # Handles boundaries gracefully
        anti_aliasing=True,  # Smooth resizing
    )
conv_EU = (cape > 500) & (shear_r > 10)
cp_EU = xr.open_dataset(scr_data+'EUR_40yr_sfc.nc').cp
cp_EU = cp_EU.sel(time=cp_EU.time.dt.year.isin(np.arange(2001,2024)))
reg_arr = xr.open_dataset(scr_data+'regions.nc')

file=np.load('/storage/homefs/mf23m219/clusters/cluster.npz')
regs=file['arr_1']
ids=file['arr_2']
region = ids[reg]
regtag = regs[reg]

reg_arr_bin = reg_arr==region
reg_size = reg_arr_bin.sum(dim=['lat','lon'],skipna=True)
conv_reg = conv_EU * reg_arr_bin
cp_reg = (conv_reg * cp_EU).max(dim=['lat','lon'],skipna=True)
conv_size = conv_reg.sum(dim=['lat','lon'],skipna=True)
conv_events = conv_size.isel(time=np.where(((conv_size>(0.25*reg_size)) & (cp_reg>0)).__xarray_dataarray_variable__.values)[0]).sortby('time',ascending=True).time

dates_diff = conv_events.diff(dim='time').dt.days.fillna(0)
datesel = np.where(dates_diff>1)[0]
first_days = conv_events.isel(time=datesel-1).time
second_days = conv_events.isel(time=datesel).time
delta_days = (second_days.time.values - first_days.time.values)/np.timedelta64(1, "D")

single_days = first_days[delta_days>3]
multi_days = first_days[delta_days<2]

rest_days = conv_size.isel(time=np.where(~((conv_size>(0.25*reg_size)) & (cp_reg>0)).__xarray_dataarray_variable__.values)[0]).sortby('time',ascending=True).time

def mw_test(x1, x2):
    """Performs Mann-Whitney U test between two 1D arrays."""

    x1 = x1[~np.isnan(x1)]  # Remove NaNs
    x2 = x2[~np.isnan(x2)]  # Remove NaNs

    if len(x1) < 2 or len(x2) < 2:  # U-test needs at least 2 values per group
        return np.nan
    
    if len(x1) == len(x2) and np.all(np.sort(x1) == np.sort(x2)):
        return np.nan  # Identical distributions → Test is invalid
    return mannwhitneyu(x1, x2, alternative="two-sided").pvalue


def generate_random_indices(time_max, num_samples):
    random_time = []
    for _ in range(num_samples):
        time = np.random.randint(5, time_max - 5)
        random_time.append(time)
    return random_time



def get_composite(res,clim,t_p,nc_p,tag,want_pca=False):
    print(t_p.size,t_p)
    coords1 = {
        'time': np.arange(-5,6,1),  # Time dimension
        'lat': clim.lat,  # Latitude dimension
        'lon': clim.lon,  # Longitude dimension
        'events': t_p.time.to_pandas(),  # Event dimension
        }

    fill1 = np.zeros([len(coords1['time']),len(coords1['lat']),len(coords1['lon']),len(coords1['events'])])
    fill1[:] = np.nan

    slices = xr.Dataset(
        {tag : (['time', 'lat', 'lon', 'events'], copy.deepcopy(fill1)),
        },
        coords1)

    a2=1
    for ttime in t_p:
        if a2%100==0: print(str(100*a2/len(t_p))+' %')
        a2+=1
        # prop = props[1]
        # time = np.datetime64(prop.time).astype("datetime64[D]")
        date = slices.isel(events=a2-2).events
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            itime = np.where(clim.time.astype("datetime64[D]")==ttime.values.astype("datetime64[D]"))[0]
        if itime.size>0: itime = itime[0]
        else: print(ttime, 'no match found'); continue

        ext = clim.isel(time=slice(itime-5,itime+6)).assign_coords(
            time=coords1['time'],lat=clim.lat,lon=clim.lon)
        # print('clim dims:',clim.dims)
        # print('ext dims:',ext.dims)
        # print('slices dims:',slices.dims)
        
        slices.loc[dict(events=date.values)] = ext


    slices = slices.sortby('events',ascending=True)
    # slices_std = slices / slices.std(dim='events')
    # print('saving slices to netcdf')
    # slices.to_netcdf(scr_data+'cookies/static_cookie_'+tag+'.nc')
    print('computing composites')
    composite = slices.mean(dim='events', skipna=True)
    print('saving composites')
    composite.to_netcdf(scr_data+'composite/static_composite_'+tag+'.nc')

    coords2 = {
        'time': np.arange(-5,6,1),  # Time dimension
        'lat': clim.lat,  # Latitude dimension
        'lon': clim.lon,  # Longitude dimension
        'events': nc_p.time.to_pandas(),  # Event dimension
        }

    fill2 = np.zeros([len(coords2['time']),len(coords2['lat']),len(coords2['lon']),len(coords2['events'])])
    fill2[:] = np.nan

    non_event = xr.Dataset(
        {tag : (['time', 'lat', 'lon', 'events'], copy.deepcopy(fill2)),
        },
        coords2)
    
    a2=1
    for ttime in nc_p:
        if a2%100==0: print(str(100*a2/len(t_p))+' %')
        a2+=1
        # prop = props[1]
        # time = np.datetime64(prop.time).astype("datetime64[D]")
        date = non_event.isel(events=a2-2).events
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            itime = np.where(clim.time.astype("datetime64[D]")==ttime.values.astype("datetime64[D]"))[0]
        if itime.size>0: itime = itime[0]
        else: print(ttime, 'no match found'); continue

        ext = clim.isel(time=slice(itime-5,itime+6)).assign_coords(
            time=coords1['time'],lat=clim.lat,lon=clim.lon)
        
        non_event.loc[dict(events=date.values)] = ext

    p_values = xr.DataArray(
        np.full((len(non_event.time), len(non_event.lat), len(non_event.lon)), np.nan),  # initialize with NaNs
        coords=[non_event.time, non_event.lat, non_event.lon],
        dims=["time", "lat", "lon"],
    )

    for t_idx in range(len(non_event.time)):
        # print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': computing MWU for timestep '+ str(t_idx))
        for lat_idx in range(len(non_event.lat)):
            for lon_idx in range(len(non_event.lon)):
                # Extract the selected and non-selected events for the current grid point
                selected_events = slices[tag].isel(time=t_idx, lat=lat_idx, lon=lon_idx)
                non_selected_events = non_event[tag].isel(time=t_idx, lat=lat_idx, lon=lon_idx)
                
                # Apply the Mann-Whitney U test and store the result in p_values
                p_values[t_idx, lat_idx, lon_idx] = mw_test(selected_events, non_selected_events)

    # non_event = non_event.sortby('events',ascending=True)
    # print('computing composites')
    # non_composite = non_event.mean(dim='events', skipna=True)
    print('saving pval')
    p_values.to_netcdf(scr_data+'composite/static_pval_'+tag+'.nc')



    if want_pca == True:

        time_size = slices.sizes["events"]
        space_size = slices.sizes["lat"] * slices.sizes["lon"]
        X = slices.sel(time=0).values.reshape(time_size, space_size)

        pca = PCA().fit(X)
        explained_variance = pca.explained_variance_ratio_
        eofs = pca.components_.reshape(pca.components_.shape[0], slices.sizes["lat"], slices.sizes["lon"])

        eof_da = xr.DataArray(
            eofs,  # Shape: (n_components, lat, lon)
            dims=["mode", "lat", "lon"],
            coords={"mode": np.arange(1, eofs.shape[0] + 1), "lat": slices.lat, "lon": slices.lon},
            name="EOFs"
        )

        pcs_da = xr.DataArray(
            pca,  # Shape: (time, n_components)
            dims=["events", "mode"],
            coords={"events": slices.events, "mode": np.arange(1, pca.shape[1] + 1)},
            name="PCs"
        )

        explained_variance_da = xr.DataArray(
            explained_variance,  # 1D array
            dims=["mode"],
            coords={"mode": np.arange(1, len(explained_variance) + 1)},
            name="explained_variance"
        )

        pca_ds = xr.Dataset({
            "EOFs": eof_da,
            "PCs": pcs_da,
            "explained_variance": explained_variance_da
        })

        # Save to NetCDF
        pca_ds.to_netcdf(scr_data + 'composite/static_composite_pca_'+tag+'.nc')

def get_composite_trend(res,clim,t_p,nc_p,tag,want_pca=False):
    print(t_p.size,t_p)
    coords1 = {
        'time': np.arange(-5,6,1),  # Time dimension
        'lat': clim.lat,  # Latitude dimension
        'lon': clim.lon,  # Longitude dimension
        'events': t_p.time.to_pandas(),  # Event dimension
        }

    fill1 = np.zeros([len(coords1['time']),len(coords1['lat']),len(coords1['lon']),len(coords1['events'])])
    fill1[:] = np.nan

    slices = xr.Dataset(
        {tag : (['time', 'lat', 'lon', 'events'], copy.deepcopy(fill1)),
        },
        coords1)

    a2=1
    for ttime in t_p:
        if a2%100==0: print(str(100*a2/len(t_p))+' %')
        a2+=1
        # prop = props[1]
        # time = np.datetime64(prop.time).astype("datetime64[D]")
        date = slices.isel(events=a2-2).events
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            itime = np.where(clim.time.astype("datetime64[D]")==ttime.values.astype("datetime64[D]"))[0]
        if itime.size>0: itime = itime[0]
        else: print(ttime, 'no match found'); continue

        ext = clim.isel(time=slice(itime-5,itime+6)).assign_coords(
            time=coords1['time'],lat=clim.lat,lon=clim.lon)
        
        slices.loc[dict(events=date.values)] = ext


    slices = slices.sortby('events',ascending=True)
    slices_ym = slices.groupby('events.year').mean('events',skipna=True)
    slices_ym = slices_ym.chunk({'lat': 50, 'lon': 50, 'year': -1})
    # print('saving slices to netcdf')
    # slices.to_netcdf(scr_data+'cookies/static_cookie_'+tag+'.nc')
    print('computing hamed-rao trend and significance')

    pval, trend = xr.apply_ufunc(
        mk_test,
        slices_ym,#numeric_time,
        input_core_dims=[['year']],#['year']],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float],
    )
    print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),': saving pvalues and trend')
    pval.to_netcdf(scr_data+'ym_pval_hr_'+tag+'.nc')
    trend.to_netcdf(scr_data+'ym_slope_hr_'+tag+'.nc')

def mk_test(da):
    """Apply Modified Mann-Kendall test to a 1D time series (single grid cell)."""
    if np.all(np.isnan(da)):
        return np.nan  # Return NaN if all values are missing
    da = da[~np.isnan(da)]
    result = mk.hamed_rao_modification_test(da,lag=1)
    
    #result = mk.original_test(da)
    return result.p, result.slope

def deaseasonalized_roll_climatology(var_da, window_size=10):

    rolling_window_size = 10
    
    rolling_mean = var_da.groupby("time.dayofyear").mean("time")

    #rolling_mean = var_da.pad(time=(rolling_window_size // 2), mode='reflect')
    rolling_mean = rolling_mean.rolling(dayofyear=rolling_window_size, center=True).mean()
    deseason_var = (var_da.groupby("time.dayofyear") - rolling_mean).assign_coords(time=var_da.time)

    return deseason_var



vars_clim = []
vars_abs_sfc = []
vars_pl = []
vars_spec = []
vars_wcb = []
vars_di = []

# if t0==0: vars_clim=[['LTG'][v0]]


file=np.load('/storage/homefs/mf23m219/clusters/cluster.npz')
arrays=file.files
labels=file['arr_0']
regs=file['arr_1']
ids=file['arr_2']
region = ids[reg]
regtag = regs[reg]
coords = {'lat': np.arange(20,60.05,0.25), 'lon': np.arange(-70,30.05,0.25)}
label_N = xr.DataArray((labels==region)*1.0,#(labels==10)*1.0+(labels==3)*1.0+(labels==7)*1.0,
                       dims=['lat','lon'], coords=coords)

t_p = pd.read_csv(scr_data+'EUR_conv_obj.csv')


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
print(len(t_p))
t_p.ilat = np.round(t_p.ilat).astype(int)
t_p.ilon = np.round(t_p.ilon).astype(int)
dtlist=[]
for year in np.arange(2000,2025):
    for month in np.arange(5,10):
        dtlist.append(str(year)+'-'+str(month).zfill(2)+'-01')

ds_s=[]
for idt,dt in enumerate(dtlist[:-11]):
    ts = pd.date_range(start=dt,end=dtlist[idt+1])
    d_t = dt[:4]+dt[5:7]
    file = glob('/storage/homefs/mf23m219/ATDNET/'+d_t+'*grid*.nc'); print(file)
    if len(file)>0:
        ds = xr.open_dataset(file[0])
        ts = ts[:len(ds.time)]
        ds = ds.assign_coords(time=ts)
        ds_s.append(ds)

dsets = xr.concat(ds_s,dim='time')['ltg_binary']

dsets_n = dsets.rename({'lat': 'lats', 'lon': 'lons'})
dsets_n = dsets_n.rename({'x': 'lat', 'y': 'lon'})
dsets_n['lon'] = ('lon', dsets.lon.values[1,:])
dsets_n['lat'] = ('lat', dsets.lat.values[:,1])

print('dsets dims:',dsets_n.dims)

deseason_dset = deaseasonalized_roll_climatology(dsets_n)
var='LTG_ANOM'
# deseason_dset.to_netcdf(scr_data+'EUR_40yr_desea_'+var+'_anom.nc')

tag = regtag+'clim'
res=0.25
tag = regtag+'_'+var+'_'+exp

get_composite(res,deseason_dset,conv_events,rest_days,'all_'+tag)
get_composite(res,deseason_dset,first_days,rest_days,'frs_'+tag)
get_composite(res,deseason_dset,single_days,rest_days,'sgl_'+tag)
get_composite(res,deseason_dset,multi_days,rest_days,'str_mlt_'+tag)
