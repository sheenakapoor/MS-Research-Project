#!/usr/bin/env python
import iris
from iris.coords import DimCoord
from iris.cube import Cube
import iris.analysis
# import iris.plot as iplt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from pyresample import image, geometry,area_config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import iris.plot as iplt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

filename = 'n216_land_sea_mask_from_um.nc'
cubes = iris.load_cube(filename)
# regrid 
target_cube = iris.load_cube("/jet/home/sheenak/tmp_ondemand_ocean_atm200005p_symlink/sheenak/nc_noon/noon_regridded/CPPin20160101120000305SVMSGE1MD_new.nc")
cubes = cubes.collapsed([cubes.coord(var_name='t'), cubes.coord(var_name='surface')],
              iris.analysis.MEAN)
lsm_regridded = cubes.regrid(target_cube, iris.analysis.Linear())
print(lsm_regridded)


# regrid plot
plt.figure(figsize=(21, 8))
iplt.pcolormesh(lsm_regridded, cmap='RdBu_r')
plt.title('regridded lsm')
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
plt.colorbar(orientation="horizontal",fraction=0.07,anchor=(1.0,0.0))
plt.savefig('lsm_regridded.png')
plt.show()

# indices_1 = argrelextrema(sp_1,  np.greater)
# minima_1 = sp_1[indices_1]
# new_pressure = np.where(land_mask==1, 110000,old_pressure)


