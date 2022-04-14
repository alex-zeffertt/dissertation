#!/usr/bin/env python3
#
# Build a social network graph representative of individuals in North Jutland, Denmark.
# The edges represent social ties along which influence over dietary habits may pass.
# 
# The graph is a networkx.Graph, i.e. an undirected graph without multiple edges between nodes.
# Edges have two weights, a "strong tie" weight which is reserved for cohabitant ties; and a
# "weak tie" weight, reserved for non-cohabitants with which individuals frequently eat (>once
# per month).
#
# The distribution of households for each household size are assumed to match that for Denmark
# as a whole and the distribution for Denmark for Jan 2022 from statistica.com is used.  Each
# household is placed at random on the map according to the geographical distribution derived
# using Facebook data and saved in NorthJutlandPopDensityGridded.pickle.  Strong tie edges are
# placed between every pair of cohabitants.
#
# We shall assume that the number of weak ties conforms to a Poisson distribution, which is
# determined entirely its mean.  Since weak ties correspond to individuals which regularly dine
# together (but do not cohabit) it is reasonable to assume that the physical distance 
# distance between them is determined by a distribution with mean 0.  We shall use a normal
# distribution, which is therefore characterized by standard_deviation sigma alone.

import pickle
import networkx as nx
import numpy as np
import sys

###### Configuration parameters #######
strong_tie_weight = 1.0
weak_tie_weight = 0.5
mean_n_weak_ties  = 3   if len(sys.argv) < 2 else int(sys.argv[1])
weak_tie_sigma_km = 1.0 if len(sys.argv) < 3 else float(sys.argv[1])

###### Datasets which we need to build a representative graph #######

# Read in population density data per km2 (z_grid)
x_grid,y_grid,z_grid = pickle.loads(open('NorthJutlandPopDensity/NorthJutlandPopDensityGridded.pickle','rb').read())
x_min, x_max = x_grid[0,0], x_grid[-1,0]
y_min, y_max = y_grid[0,0], y_grid[0,-1]
x_range = x_max - x_min
y_range = y_max - y_min

# Work out grid areas and population of North Jutland n_pop_NJ
R_km = 6371 # Radius of earth
dx_deg = x_grid[1,0]-x_grid[0,0]
dy_deg = y_grid[0,1]-y_grid[0,0]
dx_km = (np.pi*dx_deg/180)*R_km*np.cos(np.pi*y_grid[0,0]/180)
dy_km = (np.pi*dy_deg/180)*R_km
dA_km2 = dx_km*dy_km
n_pop_NJ = int(z_grid.sum()*dA_km2)

# Data for Jan 2022 from
# https://www.statista.com/statistics/582641/households-by-household-size-in-denmark/
# NB: In the original data the last line is "8 or more" but we are going to
# truncate
n_households_by_size_DK = { 1: 1107109, 2: 929014, 3: 315346, 4: 299579,
                            5: 102080,  6: 23200,  7: 6560,   8: 5403}

# Calculate equivalent for North Jutland, assuming the proportions are similar
n_pop_DK = sum([ k*v for k,v in n_households_by_size_DK.items() ])
n_households_by_size_NJ = { k: round(v*n_pop_NJ/n_pop_DK)
                            for k,v in n_households_by_size_DK.items() }

# Create a undirected graph in which each node is a person
# indexed by their position (in degrees longitude and latitude).
# edges are strong or weak ties.  Strong is interpreted as co-habitation
g = nx.Graph()

# Add the strong ties by placing households at random according to pop Density
m,n = z_grid.shape
idx = np.arange(m*n, dtype=int)
probs = z_grid.flatten() / z_grid.sum()

for k,v in n_households_by_size_NJ.items():
    household_sz = k
    n_households = v
    household_idx = np.random.choice(idx, n_households, p=probs)
    household_x_idx = household_idx//n
    household_y_idx = household_idx%n
    jitter_x = np.random.normal(0,dx_deg/3,size=n_households)
    jitter_y = np.random.normal(0,dy_deg/3,size=n_households)
    household_x = x_grid[household_x_idx,household_y_idx] + jitter_x
    household_y = y_grid[household_x_idx,household_y_idx] + jitter_y
    # house radius in km
    r_km = 0.02
    for x,y in zip(household_x,household_y):
        nodes = []
        weighted_edges = []
        for i in range(household_sz):
            x_node = x + r_km*dx_deg/dx_km * np.cos(2*np.pi*i/household_sz)
            y_node = y + r_km*dy_deg/dy_km * np.sin(2*np.pi*i/household_sz)
            nodes.append((x_node,y_node))
        for i in range(1,len(nodes)):
            for j in range(i):
                weighted_edges.append((nodes[i],nodes[j],strong_tie_weight))
        g.add_nodes_from(nodes)
        g.add_weighted_edges_from(weighted_edges)


# Add random links between households

# reuse same prob distribution within each sigma_x by sigma_y block for efficiency speedup
sigma_km = weak_tie_sigma_km
sigma_x = weak_tie_sigma_km * dx_deg/dx_km
sigma_y = weak_tie_sigma_km * dy_deg/dy_km
epsilon = 0.0001
x_blocklims = np.linspace(x_min-epsilon,x_max+epsilon,round(x_range/sigma_x))
y_blocklims = np.linspace(y_min-epsilon,y_max+epsilon,round(y_range/sigma_y))

# Choose a number of weak ties for each node
nodes = np.array(g.nodes)
node_idx = np.arange(len(nodes),dtype=int)
n_weak_ties = np.random.poisson(lam=mean_n_weak_ties,size=len(nodes))
max_n_weak_ties = max(n_weak_ties[:])

for i in range(len(x_blocklims)-1):
    x_start, x_end = x_blocklims[i], x_blocklims[i+1]
    x_midpt = (x_start + x_end)/2
    print(f"mean_n_weak_ties={mean_n_weak_ties} weak_tie_sigma_km={weak_tie_sigma_km} {i}/{len(x_blocklims)}")
    for j in range(len(y_blocklims)-1):
        y_start, y_end = y_blocklims[j], y_blocklims[j+1]
        y_midpt = (y_start + y_end)/2
        
        # create probabilities for making links to each node in NJ using normal distrib
        delta_deg = nodes - np.array((x_midpt,y_midpt))
        delta_km = delta_deg * np.array((dx_km/dx_deg, dy_km/dy_deg))
        dists_km = np.sqrt(delta_km[:,0]**2 + delta_km[:,1]**2)
        if dists_km.min() > 6*sigma_km:
            continue
        prob_unscaled = np.exp(-0.5*(dists_km/sigma_km)**2)
        prob = prob_unscaled/prob_unscaled.sum()

        # subset of nodes in this block
        inblock = \
            (x_start <= nodes[:,0]) & (nodes[:,0] < x_end) & \
            (y_start <= nodes[:,1]) & (nodes[:,1] < y_end)
        node_idx_inblock = node_idx[inblock]
        n_inblock = len(node_idx_inblock)

        weighted_edges = []
        peer_idxs = np.random.choice(node_idx,p=prob,size=(n_inblock,max_n_weak_ties))
        for idx, _peer_idxs, n_peers in zip(node_idx_inblock, peer_idxs, n_weak_ties[node_idx_inblock]):
            weighted_edges.extend([
                (tuple(nodes[idx]),tuple(nodes[peer_idx]),weak_tie_weight)
                for peer_idx in _peer_idxs[:n_peers] if peer_idx != idx
            ])
        g.add_weighted_edges_from(weighted_edges)

# Save graph
open(f'NorthJutlandSocialGraph_{mean_n_weak_ties}_{weak_tie_sigma_km}.pickle','wb').write(pickle.dumps(g))
sys.exit(0)

# Plot population density plus nodes plus links
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from shapely.geometry import Point 
from shapely.geometry.polygon import Polygon
from matplotlib import collections as mc

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

plt.ion()
fig,ax = plt.subplots()
plt.contourf(x_grid,y_grid,z_grid,10,
             vmin=min(z_grid.flatten()),
             vmax=max(z_grid.flatten()),
             cmap='coolwarm') 
polygons = pickle.loads(
    open('NorthJutlandBoundary/NorthJutlandBoundary.pickle','rb').read())
outside = Polygon(
    shell=((x_min-x_range/20,y_min-y_range/20),
           (x_max+x_range/20,y_min-y_range/20),
           (x_max+x_range/20,y_max+y_range/20),
           (x_min-x_range/20,y_max+y_range/20)),
    holes=tuple(tuple(p.boundary.coords)for p in  polygons))
# Read in polygons for 'North Jutland'
plot_polygon(plt.gca(), outside, lw=0, facecolor='white')
for p in polygons:
    plt.plot(*zip(*p.boundary.coords), lw=1, color='black')
plt.colorbar()
ax.add_collection(mc.LineCollection(g.edges, linewidths=1, color='black'))
ax.set_aspect((dy_km/dy_deg) / (dx_km/dx_deg))
plt.title("North Jutland population density per km$^2$")
