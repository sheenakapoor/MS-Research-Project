import iris
from iris.coords import DimCoord
from iris.cube import Cube
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from pyresample import image, geometry,area_config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import glob

# midday SEVIRI files from 2016
# copied files to nc_noon using:
# find /jet/home/sheenak/tmp_ondemand_ocean_atm200005p_symlink/mkhan2/SEVERI_2016/ORD46294/ \( -name "*0000305SVMSGE1MD.nc" \) -exec cp {} nc_noon \;

filename = list(glob.glob("*.nc"))

def regrid(file, file_out):
    DS = xr.open_dataset(file)
    print(file)
    cot = np.array(DS.cot[0])
    msg_area = geometry.AreaDefinition('msg_full', 'Full globe MSG image 0 degrees','msg_full',DS.CMSAF_proj4_params, 3636, 3636, DS.CMSAF_area_extent)
    msg_con_nn = image.ImageContainerNearest(cot, msg_area, radius_of_influence=50000)
    msg_con_nn.fill_value = -99
    area_def = geometry.AreaDefinition.from_extent(area_id='azores',projection='+proj=eqc',shape=(3636,3636),area_extent=(-65,25,-5,60),units='deg')
    area_con_quick = msg_con_nn.resample(area_def)
    result_data_quick = area_con_quick.image_data
    lon,lat = area_def.get_lonlats()
    crs = area_def.to_cartopy_crs()

    # plot
    plt.figure(figsize=(21, 8))
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    plt.pcolormesh(lon[0,:],lat[:,0],result_data_quick, vmin=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--') 
    plt.gca().coastlines()
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'color': 'red', 'weight': 'bold'}
    gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    plt.colorbar(orientation="horizontal",fraction=0.07,anchor=(1.0,0.0))
    fig.savefig("noon_regridded/png_regridded/"+str(file.replace(".nc", ".png")))
    plt.show()

    #cube
    longitude = DimCoord(lon[0,:], standard_name = 'longitude', units = 'degrees')
    latitude = DimCoord(lat[:,0], standard_name = 'latitude', units = 'degrees')
    my_cube = iris.cube.Cube(cot, "atmosphere_optical_thickness_due_to_cloud",  dim_coords_and_dims=[(latitude,0), (longitude,1)])

    print(my_cube)
    iris.save([my_cube], file_out, saver=iris.fileformats.netcdf.save)

for index, file in enumerate(filename):
    cubes = iris.load(file)
    regrid(file, "noon_regridded/"+str(file.replace(".nc", "_new.nc")))
