#!/usr/bin/env python3
#
# Process some population density from Meta (formerly Facebook) for North Jutland

import pickle
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Point 
from shapely.geometry.polygon import Polygon

# Read in population density for Denmark (2020) from
# https://data.humdata.org/dataset/c846d97b-4363-469a-b66b-1ac2674913b0/resource/cd2c4503-053a-4006-9d0f-efd985e2188f/download/dnk_general_2020_csv.zip
#
# dnk_general_2020.csv has first row: 'longitude', 'latitude', 'dnk_general_2020'
# where the last line represents population density, but no units are given by the source.

dat = np.genfromtxt('dnk_general_2020.csv', delimiter=',',skip_header=True)
x = dat[:,0]
y = dat[:,1]
z = dat[:,2]

# Read in polygons for 'North Jutland'
polygons = pickle.loads(open('../NorthJutlandBoundary/NorthJutlandBoundary.pickle','rb').read())
polygons.sort(key=lambda p:p.area, reverse=True)
              
# Function to determine whether point (x[i],y[i]) is in North Jutland
def isNorthJutland(coords):
    point = Point(*coords)
    for poly in polygons:
        if poly.contains(point):
            return True
    return False

# Clip the pop density data to North Jutland
# x is longitude/degress
# y is latitude/degress
# z is a measure proportional to population density 
valid = np.array(list(map(isNorthJutland, zip(x,y))))
x = x[valid]
y = y[valid]
z = z[valid]

# save raw x,y,z values for North Jutland
open('NorthJutlandPopDensityRaw.pickle','wb').write(pickle.dumps((x,y,z)))

# generate gridded and scaled population density data from x,y,z
x_min, x_max, x_range = x.min(), x.max(), x.max() - x.min()
y_min, y_max, y_range = y.min(), y.max(), y.max() - y.min()
x_grid, y_grid = np.mgrid[x_min:x_max:1000j, y_min:y_max:1000j]
z_grid = griddata((x,y), z, (x_grid, y_grid), fill_value=0.,method='linear')

# Clip to North Jutland
for i,j in zip(*z_grid.nonzero()):
    if not isNorthJutland((x_grid[i,j],y_grid[i,j])):
        z_grid[i,j] = 0.

# Work out grid areas
R_km = 6371 # Radius of earth
dx_deg = x_grid[1,0]-x_grid[0,0]
dy_deg = y_grid[0,1]-y_grid[0,0]
dx_km = (np.pi*dx_deg/180)*R_km*np.cos(np.pi*y_grid[0,0]/180)
dy_km = (np.pi*dy_deg/180)*R_km
dA_km2 = dx_km*dy_km

# Rescale z_grid to be known population density in people per km2
pop = 590322 # From http://www.statistikbanken.dk/FOLK1, April 2021
z_grid = (pop*z_grid/z_grid.sum())/dA_km2

# save gridded and scaled x_grid,y_grid,z_grid data
# This is longitude/deg, latitude/deg, popdensity/(1/km^2)
open('NorthJutlandPopDensityGridded.pickle','wb').write(
    pickle.dumps((x_grid,y_grid,z_grid)))
