#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point 
from shapely.geometry.polygon import Polygon

# Read in the polygons for 'North Jutland' from the csv files
polygons = []
for i in range(1,29):
    dat = np.genfromtxt(f'NorthJutlandPolygon{i}.csv', delimiter=',',skip_header=True)
    x = dat[:,1]
    y = dat[:,2]
    polygons.append(Polygon(zip(x,y)))

# Check we've done the right thing
for p in polygons:
    x,y = zip(*p.boundary.coords)
    plt.plot(x,y)

plt.show()


# Save polygons for 'North Jutland' to file
open('NorthJutlandBoundary.pickle','wb').write(pickle.dumps(polygons))
