#!/usr/bin/env python
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
import netCDF4
import numpy as np
from scipy.cluster.vq import *
from matplotlib import colors as c

f = netCDF4.Dataset('sp_minima_0.nc', 'r') #data for jan 1
lats = f.variables['latitude'][:]
lons = f.variables['longitude'][:]
sp = f.variables['surface_air_pressure'][:]
sp = np.nan_to_num(sp)

# Flatten image to get line of values
flatraster = sp.flatten()
flatraster.mask = False
flatraster = flatraster.data

# Create figure to receive results
fig = plt.figure(figsize=[20,7])
fig.suptitle('K-Means Clustering')

# In first subplot add original image
ax = plt.subplot(241)
ax.axis('off')
ax.set_title('surface air pressure')
original=ax.imshow(sp, cmap='rainbow', interpolation='nearest', aspect='auto', origin='lower')
plt.colorbar(original, cmap='rainbow', ax=ax, orientation='vertical')
plt.gca().invert_yaxis()
# In remaining subplots add k-means clustered images
# Define colormap
list_colors=['blue','orange', 'green', 'magenta', 'cyan', 'gray', 'red', 'yellow']
for i in range(7):
    print("Calculate k-means with ", i+2, " clusters.")
    
    #This scipy code clusters k-mean, code has same length as flattened
    # raster and defines which cluster the value corresponds to
    centroids, variance = kmeans(flatraster.astype(float), i+2)
    code, distance = vq(flatraster, centroids)
    
    #Since code contains the clustered values, reshape into SAR dimensions
    codeim = code.reshape(sp.shape[0], sp.shape[1])
    
    #Plot the subplot with (i+2)th k-means
    ax = plt.subplot(2,4,i+2)
    ax.axis('off')
    xlabel = str(i+2) , ' clusters'
    ax.set_title(xlabel)
    bounds=range(0,i+2)
    cmap = c.ListedColormap(list_colors[0:i+2])
    kmp=ax.imshow(codeim, interpolation='nearest', aspect='auto', cmap=cmap,  origin='lower')
    plt.colorbar(kmp, cmap=cmap,  ticks=bounds, ax=ax, orientation='vertical')
    plt.gca().invert_yaxis()
plt.show()

from skimage import measure

# Find contours
centroids, variance = kmeans(flatraster.astype(float), 6)
code, distance = vq(flatraster, centroids)
codeim = code.reshape(sp.shape[0], sp.shape[1])

thresholded = np.zeros(codeim.shape)
thresholded[codeim==4] = 110000
contours = measure.find_contours(thresholded, 99000)

# Display the image and plot all contours found
fig = plt.figure(figsize=[20,7])
ax = plt.subplot()
ax.set_title('surface air pressure')
original=ax.imshow(thresholded, cmap='rainbow', interpolation='nearest', aspect='auto', origin='lower')
plt.colorbar(original, cmap='rainbow', ax=ax, orientation='vertical')
plt.gca().invert_yaxis()
from scipy.spatial import distance #if based on shape was an option: more spherical ones would be chosen
for n, contour in enumerate(contours):
    dists = distance.cdist(contour, contour, 'euclidean') #chebyshev also works the same for me
    if dists.max() > 250:
        ax.fill(contour[:, 1], contour[:, 0], linewidth=2, color='black', alpha =0.6)
        print(dists.max())
        
        
print(contour)
