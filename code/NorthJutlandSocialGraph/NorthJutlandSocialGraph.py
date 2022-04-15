#!/usr/bin/env python3
#
# Build a social network graph representative of individuals in North Jutland, Denmark.
# The edges represent social ties along which influence over dietary habits may pass.
# 
# The graph is a undirected graph without multiple edges between nodes.
# Edges are either "strong ties" i.e. cohabitant ties; or "weak ties", i.e. non-cohabitants
# who eat together frequently (>once per month). 
#
# The distribution of households for each household size are assumed to match that for Denmark
# as a whole and the distribution for Denmark for Jan 2022 from statistica.com is used.  Each
# household is placed at random on the map according to the geographical distribution derived
# using Facebook data and saved in NorthJutlandPopDensityGridded.pickle.  Strong tie edges are
# placed between every pair of cohabitants.
#
# We shall assume that the number of weak ties conforms to a Poisson distribution, which is
# determined entirely by mean_n_weak_ties.  Since weak ties correspond to individuals which
# regularly dine together (but do not cohabit) it is reasonable to assume that the closer two
# individuals are geographically the more likely they are to form a weak tie.  Each such edge
# corresponds to a vector and we shall assume that the x and y components of these are independent
# random variables from normal distributions each with mean 0 and sd sigma_km.  A mathematical
# consequence is that the resulting distribution of edge lengths has a right skewed distribution
# whose modal length is also sigma_km.  However, we will need to verify that the non-uniform
# character of the population density does not significantly change this.
#
# The final graph is represented as a tuple
#
#     (coords, edges, strong_tie)
#
# where coords is an Nx2 numpy array of longitude and latitude, edges is a Ex2 array of
# indices into coords, and strong_tie is an 1xE array of bools.  The tuple is saved in the file
#
#     ./NorthJutlandSocialGraph_{mean_n_weak_ties}_{weak_tie_sigma_km}.pickle
#
# Usage:
#
# Run this file with the command line
#
#     ./NorthJutlandSocialGraph.py <mean_n_weak_ties>_<weak_tie_sigma_km>

import pickle
import numpy as np
import sys

###### Configuration parameters ########
# NB: population_thinning_factor is a debug option to speed things up
mean_n_weak_ties  = 6   if len(sys.argv) < 2 else int(sys.argv[1])
modal_weak_tie_km = 1.0 if len(sys.argv) < 3 else float(sys.argv[2])
population_thinning_factor = 1  if len(sys.argv) < 4 else float(sys.argv[3])

print(f"mean_n_weak_ties={mean_n_weak_ties} "
      f"modal_weak_tie_km={modal_weak_tie_km}")
###### Datasets which we need to build a representative graph #######

# Read in population density data per km2 (z_grid)
x_grid,y_grid,z_grid = pickle.loads(open('../NorthJutlandPopDensity/NorthJutlandPopDensityGridded.pickle','rb').read())
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
n_pop_NJ = int(z_grid.sum()*dA_km2/population_thinning_factor)

# Data for Jan 2022 from
# https://www.statista.com/statistics/582641/households-by-household-size-in-denmark/
# NB: In the original data the last line is "8 or more" but we are going to
# truncate
n_households_by_size_DK = {
    1: 1107109, 2: 929014, 3: 315346, 4: 299579,
    5: 102080,  6: 23200,  7: 6560,   8: 5403
}
# Calculate equivalent for North Jutland, assuming the proportions are similar
n_pop_DK = sum([ k*v for k,v in n_households_by_size_DK.items() ])
n_households_by_size_NJ = { k: round(v*n_pop_NJ/n_pop_DK) for k,v in n_households_by_size_DK.items() }
n_nodes = sum([ k*v for k,v in n_households_by_size_NJ.items() ])

# edge_dict is a lookup (idx1,idx2) -> bool(strong_tie) where idx1>idx2 and will be converted later on to edges
coords = np.ndarray((n_nodes, 2), dtype=float)
idxs = np.arange(n_nodes)
edge_dict = {}


# Add the strong ties by placing households at random according to pop Density
m,n = z_grid.shape
grid_idx = np.arange(m*n, dtype=int)
probs = z_grid.flatten() / z_grid.sum()

idx = 0
for k,v in n_households_by_size_NJ.items():
    household_sz = k
    n_households = v
    household_idx = np.random.choice(grid_idx, n_households, p=probs)
    household_x_idx = household_idx//n
    household_y_idx = household_idx%n
    jitter_x = np.random.normal(0,dx_deg/3,size=n_households)
    jitter_y = np.random.normal(0,dy_deg/3,size=n_households)
    household_x = x_grid[household_x_idx,household_y_idx] + jitter_x
    household_y = y_grid[household_x_idx,household_y_idx] + jitter_y
    # house radius in km
    r_km = 0.02
    for x,y in zip(household_x,household_y):
        for i in range(household_sz):
            x_node = x + r_km*dx_deg/dx_km * np.cos(2*np.pi*i/household_sz)
            y_node = y + r_km*dy_deg/dy_km * np.sin(2*np.pi*i/household_sz)
            coords[idx] = (x_node,y_node)
            for j in range(i):
                edge_dict[idx,idx-i+j] = True # strong tie
            idx += 1

# Add random links between households

# We can achieve a modal weak tie edge distance of modal_weak_tie_km
# by choosing x and y components independently and from normal distributions with mean 0 and sd
# equal to modal_weak_tie_km
sigma_km = modal_weak_tie_km

# reuse same prob distribution within each sigma_x_deg by sigma_y_deg block for efficiency speedup
sigma_x_deg = sigma_km * dx_deg/dx_km
sigma_y_deg = sigma_km * dy_deg/dy_km

# Create x and y bounds for each block
epsilon = 0.0001
x_blocklims = np.linspace(x_min-epsilon,x_max+epsilon,round(x_range/sigma_x_deg))
y_blocklims = np.linspace(y_min-epsilon,y_max+epsilon,round(y_range/sigma_y_deg))

# Choose a number of weak ties for each node
# NB: divide by 2 to account for fact this node can be a target as well as source
n_weak_ties = np.random.poisson(lam=mean_n_weak_ties/2,size=n_nodes)
max_n_weak_ties = max(n_weak_ties[:])

for i in range(len(x_blocklims)-1):
    x_start, x_end = x_blocklims[i], x_blocklims[i+1]
    x_midpt = (x_start + x_end)/2
    for j in range(len(y_blocklims)-1):
        y_start, y_end = y_blocklims[j], y_blocklims[j+1]
        y_midpt = (y_start + y_end)/2
        
        # create probabilities for making links to each node in NJ using normal distrib
        delta_deg = coords - np.array((x_midpt,y_midpt))
        delta_km = delta_deg * np.array((dx_km/dx_deg, dy_km/dy_deg))
        dists_km = np.sqrt(delta_km[:,0]**2 + delta_km[:,1]**2)
        if dists_km.min() > 6*sigma_km:
            continue
        prob_unscaled = np.exp(-0.5*(dists_km/sigma_km)**2)
        prob = prob_unscaled/prob_unscaled.sum()

        # subset of nodes in this block
        inblock = \
            (x_start <= coords[:,0]) & (coords[:,0] < x_end) & \
            (y_start <= coords[:,1]) & (coords[:,1] < y_end)
        inblock_idxs = idxs[inblock]
        n_inblock = len(inblock_idxs)

        # Randomly choose peers for each node in inblock_idxs
        peer_idxs = np.random.choice(idxs, p=prob, size=(n_inblock,max_n_weak_ties))
        for idx, _peer_idxs, n_peers in zip(inblock_idxs, peer_idxs, n_weak_ties[inblock_idxs]):
            n_added = 0
            for peer_idx in _peer_idxs:
                if n_added >= n_peers:
                    break
                lhs = max(idx, peer_idx)
                rhs = min(idx, peer_idx)
                edge = (lhs,rhs)
                if peer_idx != idx and edge not in edge_dict:
                    edge_dict[edge] = False # weak tie
                    n_added += 1

        # Display percentage complete
        percent_done = ((i+1)*len(y_blocklims) + (j+1))*100
        percent_done /= (len(x_blocklims)*len(y_blocklims))
        print(f"\r{percent_done:.1f}%", end="")
print("")

# Save graph
edges = np.array(list(edge_dict.keys()), dtype=int)
strong_tie = np.array([ edge_dict[i,j] for (i,j) in edges ], dtype=bool)

if 1:
    open(f'NorthJutlandSocialGraph_{mean_n_weak_ties}_{modal_weak_tie_km}.pickle','wb').write(pickle.dumps((coords, edges, strong_tie)))
else:
    # debug: plot just the strong ties
    sys.path.append('..')
    import NorthJutlandPopDensity
    import matplotlib.pyplot as plt
    from matplotlib import collections as mc
    NorthJutlandPopDensity.plot(colorbar=False)
    ax = plt.gca()
    plt.scatter(coords[:,0],coords[:,1],marker='.',
                linewidths=1,color='black')
    ax.add_collection(mc.LineCollection(coords[edges[strong_tie]],
                                        linewidths=1, colors='black'))
    ax.add_collection(mc.LineCollection(coords[edges[strong_tie==False]],
                                        linewidths=1, colors='black', alpha=0.5))
