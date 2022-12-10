#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.basemap import Basemap
from pylab import rcParams
import sklearn.utils
import netCDF4 as nc
import xarray as xr
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pyresample import image, geometry,area_config
import matplotlib.ticker as mticker
from sklearn.neighbors import NearestNeighbors

ds = xr.open_dataset('sp_minima_0.nc')
sp_DS = ds.to_dataframe()
sp_DS.reset_index(inplace=True) 
sp_DS=sp_DS.iloc[::10]
DS=xr.open_dataset('CPPin20160101120000305SVMSGE1MD_new.nc')
cot_DS=DS.to_dataframe()
cot_DS.reset_index(inplace=True)
cot_DS=cot_DS.iloc[::10]
cot_DS=cot_DS[cot_DS['atmosphere_optical_thickness_due_to_cloud'] !=0]
cot_DS = pd.concat([cot_DS, sp_DS['surface_air_pressure']], axis=1).reindex(cot_DS.index) #later i should rename to result to avoid confusion!
cot_DS = cot_DS[cot_DS['surface_air_pressure'] !=110000]
print(cot_DS)
print ("Shape of the DataFrame: ", cot_DS.shape)
print(cot_DS.isna().sum())
cot_DS.dropna(subset=['atmosphere_optical_thickness_due_to_cloud'], inplace=True)
print ("After dropping rows that contains NaN on COT column: ", cot_DS.shape)
llon=-65
ulon=0
llat=25
ulat=60

cot_DS = cot_DS[(cot_DS['longitude'] > llon) & (cot_DS['longitude'] < ulon) & 
                        (cot_DS['latitude'] > llat) &(cot_DS['latitude'] < ulat)]

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
my_map.drawlsmask(land_color='orange', ocean_color='skyblue')
my_map.bluemarble()
# To collect data based on stations        

xs,ys = my_map(np.asarray(cot_DS.longitude), np.asarray(cot_DS.latitude))
cot_DS['xm']= xs.tolist()
cot_DS['ym'] =ys.tolist()

weather_df_clus_temp = cot_DS[["longitude","latitude","atmosphere_optical_thickness_due_to_cloud", "surface_air_pressure"]]
weather_df_clus_temp = StandardScaler().fit_transform(weather_df_clus_temp)
cot_df=np.asarray(weather_df_clus_temp)

#############
#Selecting epsilon with KNearest help when considering points at 50 step
#############
# neighbors = NearestNeighbors(n_neighbors=4)
# neighbors_fit = neighbors.fit(cot_df)
# distances, indices = neighbors_fit.kneighbors(cot_df)
# distances = np.sort(distances, axis=0)
# distances = distances[:,1]
# plt.plot(distances)
# plt.show()
##################

print('clustering in progress... please be patient!')

db = DBSCAN(eps=0.2,min_samples=300).fit(weather_df_clus_temp)  #0.1,8 gave too many clusters. try more min samples. 0.2,100 gave less clusters and some clarity. 0.25, 80 is good.
labels = db.labels_
print (labels[500:560])
cot_DS["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))
print(clusterNum)

print('making figure')

rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
my_map.drawlsmask(land_color='orange', ocean_color='skyblue')
my_map.etopo()

# To create a color map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))


#Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = cot_DS[cot_DS.Clus_Db == clust_number]                    
    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 15, alpha = 0.1)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number+1), fontsize=20, color='red',)
        print ("Cluster "+str(clust_number+1)+', COT, SP resp: '+ str(np.mean(clust_set.atmosphere_optical_thickness_due_to_cloud))+ ', '+ str(np.mean(clust_set.surface_air_pressure)))
plt.title(r"COT (1): $ \epsilon = 0.3$", fontsize=13)        
plt.savefig("etopo2_cluster300.png", dpi=300)    

