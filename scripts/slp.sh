#!/usr/bin/env python
import iris
from iris.coords import DimCoord
from iris.cube import Cube
import iris.analysis
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from pyresample import image, geometry,area_config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# import glob
from scipy.signal import argrelextrema
from scipy.ndimage.filters import minimum_filter

# midday SLP data
filename = 'sealevelpressure2016.nc'
data = xr.open_dataset(filename)
data = data.reindex(latitude=data.latitude[::-1]) 
print(data)

# land sea mask
lsmcube = iris.load_cube('n216_land_sea_mask_from_um.nc')
lsm_data = xr.open_dataset('n216_land_sea_mask_from_um.nc')
land_mask = np.array(lsm_data.lsm)

# regrid 
target_cube = iris.load_cube("/jet/home/sheenak/tmp_ondemand_ocean_atm200005p_symlink/sheenak/nc_noon/noon_regridded/CPPin20160101120000305SVMSGE1MD_new.nc")
lsmcube = lsmcube.collapsed([lsmcube.coord(var_name='t'), lsmcube.coord(var_name='surface')], iris.analysis.MEAN)
lsm_regridded = lsmcube.regrid(target_cube, iris.analysis.Linear())
iris.save([lsm_regridded], 'lsm_regridded.nc', saver=iris.fileformats.netcdf.save)
lsm_regridded_data = xr.open_dataset('lsm_regridded.nc')
masked_array = np.array(lsm_regridded_data.lsm)

# creating separate arrays for data of each day (366 arrays)
sp = np.array(data.sp) #shape = (366,261,501)
spi = [sp[i,:,:] for i in range(366)]

for i,j in enumerate(sp):
    # create cube
    longitude = DimCoord(np.linspace(-100, 25, 501), standard_name = 'longitude', units = 'degrees')
    latitude = DimCoord(np.linspace(0, 65, 261), standard_name = 'latitude', units = 'degrees')
    print(spi[i])
    sp_cube = iris.cube.Cube(spi[i], "surface_air_pressure",  dim_coords_and_dims=[(latitude,0), (longitude,1)])
    iris.save([sp_cube], 'sp_'+str(i)+'.nc', saver=iris.fileformats.netcdf.save)
    
    # interpolate all sp cubes
    sample_points = [('longitude', np.linspace(-100, 25, 3636)), ('latitude',  np.linspace(0, 65, 3636))]
    result = sp_cube.interpolate(sample_points, iris.analysis.Linear()) # was plotting result earlier
    
    # regrid
    sp_regridded = result.regrid(target_cube, iris.analysis.Linear())
    iris.save([sp_regridded], 'sp_regridded_'+str(i)+'.nc', saver=iris.fileformats.netcdf.save)
    sp_regridded_data = xr.open_dataset('sp_regridded_'+str(i)+'.nc')
    
    # mask regridded sp
    sp_array = np.array(sp_regridded_data.surface_air_pressure)
    sp_masked = np.where(masked_array == 1, 110000, sp_array)
    longitude = DimCoord(sp_regridded_data.longitude, standard_name = 'longitude', units = 'degrees')
    latitude = DimCoord(sp_regridded_data.latitude, standard_name = 'latitude', units = 'degrees')
    sp_masked_cube = iris.cube.Cube(sp_masked, "surface_air_pressure",  dim_coords_and_dims=[(latitude,0), (longitude,1)])
    iris.save([sp_masked_cube], 'sp_minima/sp_minima_nc/sp_minima_'+str(i)+'.nc', saver=iris.fileformats.netcdf.save)
    
    spmc_data = xr.open_dataset('sp_minima/sp_minima_nc/sp_minima_'+str(i)+'.nc')
    spmc_masked_array = np.array(spmc_data.surface_air_pressure)
    print("spmc_masked_array")
    print(spmc_masked_array.shape)
    
    # finding the minima
    minima = (spmc_masked_array == minimum_filter(spmc_masked_array, size=(3636,300), mode='nearest'))
    index = np.where(1==minima)
    lon = np.array(index)[1, :]
    lat = np.array(index)[0, :]
    
    # regrid plot
    plt.figure(figsize=(21, 8))
    iplt.pcolormesh(sp_masked_cube, cmap='RdBu_r')
    plt.scatter(sp_regridded_data.longitude[lon], sp_regridded_data.latitude[lat], marker='*', color='cyan', s=100)
    plt.title('regridded and masked sea level pressure')
    ax = plt.gca()
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    ax.set_xlabel("")
    ax.set_ylabel("")
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'color': 'red', 'weight': 'bold'}
    gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    plt.savefig('sp_minima/sp_minima_plots/sp_minima_'+str(i)+'.png')
    plt.show()
    


