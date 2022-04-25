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

def plot():

    polygons = pickle.loads(
        open(f'{dir_path}/../NorthJutlandBoundary/NorthJutlandBoundary.pickle','rb').read())

    # Work out grid areas
    R_km = 6371 # Radius of earth
    dx_deg = 1.0
    dy_deg = 1.0
    y_deg = polygons[0].boundary.coords[0][1]
    dx_km = (np.pi*dx_deg/180)*R_km*np.cos(np.pi*y_deg/180)
    dy_km = (np.pi*dy_deg/180)*R_km

    # Plot population density
    plt.ion()
    fig = plt.gcf()
    ax = plt.gca()
    for p in polygons:
        plt.plot(*zip(*p.boundary.coords), lw=1, color='black')
    ax.set_aspect((dy_km/dy_deg) / (dx_km/dx_deg))
    plt.title("North Jutland")

if __name__ == '__main__':
    plot()
