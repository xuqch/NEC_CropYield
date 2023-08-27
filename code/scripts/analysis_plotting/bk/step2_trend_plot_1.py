# -*- coding: utf-8 -*-
"""
A libray with Python functions for  calculating trend with p-value, using [scipy].stats.linregress
****
"""
__author__ = "Zhongwang Wei / zhongwang007@gmail.com"
__version__ = "0.1"
__release__ = "0.1"
__date__ = "Oct 2020"

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os, sys
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.stats import linregress
from numba import jit #  Speedup for python functions
from matplotlib import colors
import cartopy .crs as ccrs
from pylab import rcParams
import matplotlib
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

### Plot settings
font = {'family' : 'DejaVu Sans'}
#font = {'family' : 'Myriad Pro'}
matplotlib.rc('font', **font)

params = {'backend': 'ps',
          'axes.labelsize': 12,
          'grid.linewidth': 0.2,
          'font.size': 15,
          'legend.fontsize': 12,
          'legend.frameon': False,
          'xtick.labelsize': 12,
          'xtick.direction': 'out',
          'ytick.labelsize': 12,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
rcParams.update(params)



def calc_trend(x,y):
    co = np.count_nonzero(~np.isnan(x))
    if co < 4:
        return -9999
    slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(x)), x)
    return slope #*100. #,p_value

def trend(x,y,dim='year'):
    return xr.apply_ufunc(
        calc_trend, x ,y,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
        )
def plot_trend(ds,path,var,ssp):
    """
    Plots the landsat stack for a given route
    """
    ds= ds.where(ds>-10, drop=True)
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    trend= ds.variables['trend'][:,:]
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Robinson())
    plt.contourf(lons, lats, trend)
    #cax = divider.append_axes("bottom", size="5%", pad=0.05)
    plt.colorbar(shrink = 1.0, spacing='uniform',orientation='horizontal',label='Annual trend',fraction=0.046, pad=0.04)
    #plt.colorbar(shrink = 0.6,cax=cax,label='Annual trend')
    plt.savefig('%s/%s_%s_trend.eps'%(path,var,ssp),  format='eps',dpi=400)
    #plt.show()
def save_xarray(outpath,arr,rid):
    """
    Save output
    """
    arr.to_netcdf( os.path.join(outpath,'trend_%s.nc' % (rid) ),
                  encoding = {'trend': {'dtype': 'float32',
                                        'zlib': True,
                                        'complevel': 6,
                                        '_FillValue': -9999.0} })
if __name__=='__main__':
    import glob, os, shutil
    #define the directory
    #pathin_China      = '../ssp245_China_preprocessed'
    #pathout_China     = '../ssp245_China_trend'
    models=['ACCESS-CM2','BCC_CSM2_MR','CanESM5']
    models=['BCC_CSM2_MR']
    for model in models:
        scratch           = ('/Volumes/RBL/Processed/%s/scratch/'%(model))
        pathout_Dongbei   = ('/Volumes/RBL/Processed/%s/trend'%(model))
        #deleting the exist directory
        #if os.path.isdir(pathout_China):
        #shutil.rmtree(pathout_China)
        if os.path.isdir(pathout_Dongbei):
            shutil.rmtree(pathout_Dongbei)
        if os.path.isdir(scratch):
            shutil.rmtree(scratch)
        # re-create the directory
        #os.mkdir(pathout_China)
        os.makedirs(pathout_Dongbei)
        os.makedirs(scratch)
        ssps = ['126','245','370','585']
        for ssp in ssps:
            pathin_Dongbei    = ('/Volumes/RBL/Processed/%s/ssp%s'%(model,ssp))
            print(pathin_Dongbei)
            #set variables
            #Vars = ['hfls', 'hfss','hur', 'hus', 'huss', 'mrro', 'mrso', 'pr', 'prsn', 'rlds', 'rlus', 'rlut','rsds','rsus','snc','snw','tslsi']
            Vars =['hfls','hfss','hurs', 'mrro', 'pr', 'prsn', 'rsds','tas']
            for Var in Vars:
                os.chdir("%s"%(pathin_Dongbei))
                #for Dongbei case
                for SubVar in glob.glob(("%s_*.nc")%(Var)):
                    print(SubVar)
                    VarFile=('%s/%s'%(pathin_Dongbei,SubVar))
                    with xr.open_dataset(VarFile) as ds:
                        ds=ds['%s'%(Var)]
                        ds_annual=ds.resample(time='1Y').mean()
                        #ds_annual.to_netcdf('%s/%s_annual'%(scratch,Var),engine='netcdf4')
                        ds2 = ds_annual.pipe(lambda x: x).where(ds_annual > -3000.0, drop=True)
                        x = xr.DataArray(np.arange(len(ds2['time']))+1, dims='time',coords={'time': ds2['time']})
                        r2 = trend(ds2, x,'time').compute()  
                        r2 = r2.to_dataset(name="trend") # Convert to dataset
                        plot_trend(r2,pathout_Dongbei,Var,ssp)
                        save_xarray(pathout_Dongbei,r2,Var)
                        #ds_spatialmean=ds.groupby('time').mean()
                
                

        

        
                    