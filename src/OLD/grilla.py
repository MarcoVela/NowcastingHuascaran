from netCDF4 import Dataset
import numpy as np

file = Dataset('dat_new/flash_density_20210101-001500.nc')
lati = file.variables['latitude'][:]
loni = file.variables['longitude'][:]
dd=file.variables['density'][:]
laa=[];loo=[];densi=[]
ki=4 #*2km=8km de grilla
for i in range(0,dd.shape[0]-ki+1,ki):
    laa.append(np.ma.median(lati[i:i+ki]))
    ee=[]; loo=[]
    for j in range(0,dd.shape[1]-ki+1,ki):
        loo.append(np.ma.median(loni[j:j+ki]))
        ee.append((np.ma.sum(dd[i:i+ki,j:j+ki])))
    densi.append(ee)
deni=np.array(densi) #resultado en grilla 8km