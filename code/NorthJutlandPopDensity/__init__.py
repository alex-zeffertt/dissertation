#!/usr/bin/env python3
#
# graph.py plot the population density of North Jutland

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from shapely.geometry import Point 
from shapely.geometry.polygon import Polygon

if __name__ != '__main__':
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
else:
    dir_path = '.'

# Plots a Polygon to pyplot `ax`
def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

def plot(colorbar=True):
    # Read in population density data per km2 (z_grid)
    x_grid,y_grid,z_grid = pickle.loads(open(f'{dir_path}/NorthJutlandPopDensityGridded.pickle','rb').read())
    x_min, x_max = x_grid[0,0], x_grid[-1,0]
    y_min, y_max = y_grid[0,0], y_grid[0,-1]
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Work out grid areas
    R_km = 6371 # Radius of earth
    dx_deg = x_grid[1,0]-x_grid[0,0]
    dy_deg = y_grid[0,1]-y_grid[0,0]
    dx_km = (np.pi*dx_deg/180)*R_km*np.cos(np.pi*y_grid[0,0]/180)
    dy_km = (np.pi*dy_deg/180)*R_km

    # Plot population density
    plt.ion()
    fig,ax = plt.subplots()
    plt.contourf(x_grid,y_grid,z_grid,10,
                 vmin=min(z_grid.flatten()),
                 vmax=max(z_grid.flatten()),
                 cmap='coolwarm')
    polygons = pickle.loads(
        open(f'{dir_path}/../NorthJutlandBoundary/NorthJutlandBoundary.pickle','rb').read())
    outside = Polygon(
        shell=((x_min-x_range/20,y_min-y_range/20),
               (x_max+x_range/20,y_min-y_range/20),
               (x_max+x_range/20,y_max+y_range/20),
               (x_min-x_range/20,y_max+y_range/20)),
        holes=tuple(tuple(p.boundary.coords) for p in  polygons))
    plot_polygon(plt.gca(), outside, lw=0, facecolor='white')
    for p in polygons:
        plt.plot(*zip(*p.boundary.coords), lw=1, color='black')
    if colorbar:
        plt.colorbar()
    ax.set_aspect((dy_km/dy_deg) / (dx_km/dx_deg))
    plt.title("North Jutland population density per km$^2$")

if __name__ == '__main__':
    plot()
