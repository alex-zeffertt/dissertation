#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
import sys
from matplotlib import collections as mc
import matplotlib.pyplot as plt
from IPython.terminal.embed import InteractiveShellEmbed
from scipy.interpolate import griddata
sys.path.append('..')
import NorthJutlandBoundary

# Return the logit normal pdf for x (which may be an array or a scalar)
def logit_normal_pdf(x,mu,sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*\
        (1/(x*(1-x)))*np.exp(-(np.log(x/(1-x)))**2/(2*sigma**2))

# Run the model using the parameters given
# Returns the tuple (X,Y,X_vs_time,Y_vs_time), where
#
# X: numpy array with a category indicating the social ties to reducers for each individual
#     0: no ties
#     1: only weak ties
#     2: 1-2 strong ties, no weak ties
#     3: 1-2 strong ties + weak ties
#     4: 3+ strong ties
#
# Y: numpy array with a stage of change category per individual
#     0 : NO intention
#     1 : Intention
#     2 : Reducer
#
# X_vs_time: n_timesteps x 5 array summarizing numbers in each social tie category at each timestep
#
# Y_vs_time: n_timesteps x 3 array summarizing numbers in stage category at each timestep

def run_model(coords, edges, strong_tie,          # Defines social graph (see ../NorthJutlandSocialGraph/README)
              awareness_pc, facility_pc,          # Global parameters affecting CA model
              p_update_logit_normal_sigma,        # Describes the distribution of resistance to change
              n_timesteps,                        # How much simulated time to run model for
              randomize_beta = False):

    # Calculate P[i,j] := P(Y = i | X = j) where
    #
    # X is the categorical explanatory variable:
    #
    #   description                                    j
    # * 0 strong ties and 0 weak ties  (reference)     0
    # * 0 strong ties and 1-2 weak ties                1
    # * 1-2 strong ties, 0 weak ties                   2
    # * 1-2 strong ties, >0 weak ties                  3
    # * 3+ strong ties                                 4
    n_tie_categories = 5
    
    #
    # and Y is the categorical response variable
    #
    #   description                                    i
    # * NO intention to reduce meat intake (reference) 0
    # * Intention to reduce meat intake                1
    # * Meat reducer                                   2
    n_diet_categories = 3

    #
    # See calcProbs for explanation of derivation
    
    # From H&L paper
    beta = np.array([
        [0.0, 0.0,  0.0,  0.0,  0.0],
        [0.0, 0.69, 2.78, 3.81, 0.65],   # Intention vs NO intention
        [0.0, 0.73, 2.54, 3.19, 3.30]])  # Reducer   vs NO intention

    if randomize_beta:
        # 95% CI for beta (NB: beta is the geometric mean of min and max)
        beta_min = np.array([
            [0.0, 0.0,  0.0,  0.0,  0.0],
            [0.0, 0.27, 1.47, 1.70, 0.16],
            [0.0, 0.36, 1.55, 1.65, 1.56]])
        beta_max = np.array([
            [0.0, 0.0,  0.0,  0.0,  0.0],
            [0.0, 1.81, 5.26, 8.57, 2.65],
            [0.0, 1.45, 4.16, 6.17, 6.98]])

        # Calculate sigma for each log(beta[i,j]) distribution
        log_beta_sigma = np.zeros(beta.shape, dtype=float)
        log_beta_sigma[1:,1:] = (np.log(beta_max[1:,1:]) - \
                                 np.log(beta[1:,1:]))/1.96

        # Choose a random beta from the distributions for beta[i,j]
        log_beta_delta = np.random.normal(np.zeros(15),
                                          log_beta_sigma.flatten(),15)
        log_beta_delta.shape = (3,5)
        beta *= np.exp(log_beta_delta)
    
    # Calculate the NOintention:intention:reducer split for case X = 0 (no ties)
    pop_not_intending_with_no_ties = 100 - awareness_pc
    pop_reducing_with_no_ties = awareness_pc * facility_pc / 100
    pop_intending_with_no_ties = 100 - \
        pop_not_intending_with_no_ties - \
        pop_reducing_with_no_ties
    
    # Calculate P[:,:] as per calcProbs
    percent = [pop_not_intending_with_no_ties,
               pop_intending_with_no_ties,
               pop_reducing_with_no_ties]
    C = np.log(np.array([percent])/percent[0])
    C = C.transpose()
    P = np.exp(C + beta)/np.exp(C + beta).sum(axis=0)

    # Provide a quick lookup for finding peers
    _, peers_count = np.unique(edges.flatten(), return_counts=True)

    # peer_lookup array
    M = coords.shape[0]
    N = peers_count.max()
    peers = np.ones((M,N),dtype=int) * -1
    peers_strong = np.zeros((M,N),dtype=int)

    n_peers_found = np.zeros(M,dtype=int)
    for (idx1,idx2),strong in zip(edges,strong_tie):
        n_peers_found_idx1 = n_peers_found[idx1]
        n_peers_found_idx2 = n_peers_found[idx2]
        peers[idx1,n_peers_found_idx1] = idx2
        peers[idx2,n_peers_found_idx2] = idx1
        peers_strong[idx1,n_peers_found_idx1] = strong
        peers_strong[idx2,n_peers_found_idx2] = strong
        n_peers_found[idx1] = n_peers_found_idx1 + 1
        n_peers_found[idx2] = n_peers_found_idx2 + 1

    peers_present = peers >= 0
    
    # state lookup arrays

    # X is the explanatory variable - i.e. the number and type of ties
    # We start off by assuming no ties at all (X = 0)
    X = np.zeros(M, dtype=int)
    
    # Y is the response variable - i.e. NO intention (0), Intention (1), or Reducer (2)
    # Initialize this randomly given no one has ties to reducers
    Y = np.random.choice(n_diet_categories, M, p=P[:,0])
    
    # Different individuals show different resistance to change.
    # We characterize this by assigning to each idx a probability
    # that it's state Y[idx] will be updated in one time step.
    # Thus if p_update[idx] = 0 then Y[idx] will never change, and if
    # p_update[idx] = 0.5 then Y[idx] will be updated with a probability of 1/2
    # in any given time step.  (Even if it is updated Y[idx] may be updated
    # with the same value, since the value selected will be random with probabilities
    # based on X[idx].)
    # We assume that logit(p_update) is normally distributed with mean of 0
    # (since changing the mean is equivalent to just altering the interpretation
    # of the time step) and s.d. p_update_logit_normal_sigma.
    p_update_vals = np.linspace(0.00001,0.99999,1000)
    p_update_probs = logit_normal_pdf(p_update_vals, 0, p_update_logit_normal_sigma)
    p_update_probs /= p_update_probs.sum()
    p_update = np.random.choice(p_update_vals, M, p=p_update_probs)

    # Arrays for saving changes in populations over time, for latter plotting
    X_vs_time = np.zeros((n_timesteps,n_tie_categories),dtype=int)
    Y_vs_time = np.zeros((n_timesteps,n_diet_categories),dtype=int)

    # The main loop:
    # every time step update X and then update Y then update plot
    for tstep in range(n_timesteps):

        # Log X and Y stats
        for i in range(n_diet_categories):
            Y_vs_time[tstep,i] = (Y == i).sum()
        for j in range(n_tie_categories):
            X_vs_time[tstep,j] = (X == j).sum()

        peer_states = Y[peers]
    
        n_ties = ((peer_states == 2) & peers_present).sum(axis=1)
        n_strong_ties = ((peer_states == 2) & peers_present & peers_strong).sum(axis=1)
        n_weak_ties = n_ties - n_strong_ties
        
        # Update tie categories X
        X[(n_ties == 0)] = 0
        X[(n_strong_ties == 0) & (n_weak_ties >= 1)] = 1
        X[(n_strong_ties >= 1) & (n_strong_ties <= 2) & (n_weak_ties == 0)] = 2
        X[(n_strong_ties >= 1) & (n_strong_ties <= 2) & (n_weak_ties >= 1)] = 3
        X[(n_strong_ties >= 3)] = 4

        # Update diet categories Y
        for j in range(n_tie_categories):
            # Find indices of individuals in with this categories of ties to reducers
            idxs = (X == j).nonzero()[0]

            # Consider resistance to change: some will not consider update this time
            do_update = np.random.binomial(1,p_update[idxs])
            idxs = idxs[do_update == 1]

            # Remainers change diet randomly using probs for social n/w categegory j
            prob_diets = P[:,j]
            Y[idxs] = np.random.choice(n_diet_categories, len(idxs), p=prob_diets)

    # Return the tuple
    return (X,Y,X_vs_time,Y_vs_time)

# Plot changes in populations sizes against time
# rows in X_vs_time, Y_vs_time are indexed by timestep.
# columns in X_vs_time are the different categories indicating numbers and types of social ties to meat reducers
# columns in Y_vs_time are the different stages of change in terms of reducing meat consumption
def plot_vs_time(X_vs_time, Y_vs_time):
    assert X_vs_time.shape[0] == Y_vs_time.shape[0]
    assert X_vs_time.shape[1] == 5
    assert Y_vs_time.shape[1] == 3
    n_timesteps = X_vs_time.shape[0]
    
    # Plot Y categories vs time
    plt.figure()
    t = np.arange(n_timesteps)
    plt.plot(t,Y_vs_time[t,0], '-', label='NO intention', color='black')
    plt.plot(t,Y_vs_time[t,1], ':', label='Intention', color='black')
    plt.plot(t,Y_vs_time[t,2], '--', label='Reducer', color='black')
    plt.xticks(t[::len(t)//5])
    plt.xlabel('timestep')
    plt.yticks([0,100e3,200e3,300e3,400e3],['0','100k','200k','300k','400k']) 
    plt.ylabel('frequency')
    plt.title('Modelled dietary changes over time')
    plt.legend()

    # Plot X categories vs time
    plt.figure()
    t = np.arange(n_timesteps)
    c=['no ties','only weak ties','1-2 strong, no weak',
       '1-2 strong + weak','3+ strong']
    plt.plot(t,X_vs_time[t,0], '-',  label=c[0], color='black')
    plt.plot(t,X_vs_time[t,1], '--', label=c[1], color='black')
    plt.plot(t,X_vs_time[t,2], '-.', label=c[2], color='black')
    plt.plot(t,X_vs_time[t,3], ':',  label=c[3], color='black')
    plt.plot(t,X_vs_time[t,4], '--', label=c[4], color='black', dashes=(2,.8,8,.8))
    plt.xticks(t[::len(t)//5])
    plt.xlabel('timestep')
    plt.yticks([0,200e3,400e3,600e3],['0','200k','400k','600k']) 
    plt.ylabel('frequency')
    plt.title('Modelled changes in social ties to meat reducers over time')
    plt.legend()


# WARNING: SLOW
# coords: Mx2 numpy array with location (longitude_deg,latitude_deg) of each individual
# Y: M array with stage of change category for each individual 0=>NO intention, 1=>intention, 2=>reducer
def plot_scatter_diagram_on_map(coords,Y):
    # plot every point color coded
    plt.figure()
    NorthJutlandBoundary.plot()
    colormap = np.array(['red','yellow','green'])
    idxs = np.arange(M,dtype=int)
    plt.scatter(coords[idxs,0],coords[idxs,1],
                linewidths=1, c=colormap[Y[idxs]], alpha=0.5)

# Graphs showing how no intention, intention, and reducer are distributed
# Both as global histograms and geographically
# coords: Mx2 numpy array with location (longitude_deg,latitude_deg) of each individual
# Y: M array with stage of change category for each individual 0=>NO intention, 1=>intention, 2=>reducer
def plot_stages_of_change_distribution(coords,Y):
    x_grid,y_grid,z_grid = pickle.loads(
        open('../NorthJutlandPopDensity/NorthJutlandPopDensityGridded.pickle',
             'rb').read())
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

    # make grid squares larger
    divs = 100
    _x_grid, _y_grid = np.mgrid[x_min:x_max:divs*1j, y_min:y_max:divs*1j]
    _z_grid = griddata((x_grid.flatten(),y_grid.flatten()),
                       z_grid.flatten(), (_x_grid, _y_grid),
                       fill_value=0.,method='linear')
    _dx_deg = _x_grid[1,0]-_x_grid[0,0]
    _dy_deg = _y_grid[0,1]-_y_grid[0,0]
    _dx_km = (np.pi*_dx_deg/180)*R_km*np.cos(np.pi*_y_grid[0,0]/180)
    _dy_km = (np.pi*_dy_deg/180)*R_km
    _dA_km2 = _dx_km*_dy_km

    _n_no_intention = np.zeros((divs,divs),dtype=int)
    _n_intention = np.zeros((divs,divs),dtype=int)
    _n_reducer = np.zeros((divs,divs),dtype=int)
    _n = np.zeros((divs,divs),dtype=int)
    for idx in range(M):
        x,y  = coords[idx]
        i = round((x - x_min)/_dx_deg)
        j = round((y - y_min)/_dy_deg)
        _n[i,j] += 1
        state = Y[idx]
        if state == 0:
            _n_no_intention[i,j] += 1
        elif state == 1:
            _n_intention[i,j] += 1
        else:
            _n_reducer[i,j] += 1

    _n_no_intention_pc = np.zeros((divs,divs),dtype=float)
    _n_intention_pc = np.zeros((divs,divs),dtype=float)
    _n_reducer_pc = np.zeros((divs,divs),dtype=float)
    nz = _n.nonzero()
    _n_no_intention_pc[nz] = 100.* _n_no_intention[nz] / _n[nz]
    _n_intention_pc[nz]    = 100.* _n_intention[nz]    / _n[nz]
    _n_reducer_pc[nz]      = 100.* _n_reducer[nz]      / _n[nz]

    plt.figure()
    bins=100
    plt.hist(_n_no_intention_pc[nz], bins=bins, alpha=0.5, label='NO intention')
    plt.hist(_n_intention_pc[nz], bins=bins, alpha=0.5, label='Intention')
    plt.hist(_n_reducer_pc[nz], bins=bins, alpha=0.5, label='Reducer')
    plt.xlabel('% of grid square')
    plt.ylabel('frequency')
    plt.title(f'Histograms of %ages in {len(nz[0])} non-empty grid squares')
    plt.legend()

    plt.figure()
    NorthJutlandBoundary.plot()
    plt.contourf(_x_grid,_y_grid,_n_no_intention_pc,10,vmin=0,vmax=100,cmap='coolwarm')
    plt.title('% population in state "NO intention" by area')
    plt.colorbar()

    plt.figure()
    NorthJutlandBoundary.plot()
    plt.contourf(_x_grid,_y_grid,_n_intention_pc,10,vmin=0,vmax=100,cmap='coolwarm')
    plt.title('% population in state "Intention" by area')
    plt.colorbar()

    plt.figure()
    NorthJutlandBoundary.plot()
    plt.contourf(_x_grid,_y_grid,_n_reducer_pc,10,vmin=0,vmax=100,cmap='coolwarm')
    plt.title('% population in state "Reducer" by area')
    plt.colorbar()

# TODO inhomogenous households

# TODO bar plots of highly connected individuals (weak links)

# TODO bar plots of highly connected individuals (strong links)
            
if __name__ == '__main__':

    # Read Parameters from command line
    parser = argparse.ArgumentParser(description='CA model for spread of reductions in meat consumption in North Jutland, Denmark\n'\
                                     'Outputs population distributions X and Y at final step, where\n'\
                                     'X = [<#no ties>,<#only weak ties>,<#1-2 strong, no weak>, <#1-2 strong + weak>,<#3+ strong>]\n'\
                                     'Y = [<#NO intention>,<#Intention>,<#Reducer>]')
    parser.add_argument('-m','--mean_n_weak_ties', type=float, default=6.0,
                        help='mean number of ties to co-diners outside of household')
    parser.add_argument('-M','--modal_weak_tie_km', type=float, default=1.0,
                        help='most common distance to co-diner outside of household')
    parser.add_argument('-a', '--awareness_pc', type=float, default=30.0,
                        help='%% of those with no ties to reducers who have at least an intention to reduce')
    parser.add_argument('-f', '--facility_pc', type=float, default=30.33333,
                        help='%% of those with no ties to reducers and at least an intention to reduce who are reducers')
    parser.add_argument('-s','--p_update_logit_normal_sigma', type=float, default=0.5,
                        help='describes how resistance to change is distributed'
                        ', i.e. p_update[idx] is the proportion of timesteps that person identified by idx takes part in'
                        '.  1 indicates every step, 0 indicates no steps.  logit(p_update) is normally distributed with mean 0.5.  This variable is the sd.:'
                        '0-1.5 => p_update is unimodal, 1.5 => p_update is almost flat, >1.5 => p_update is bimodal')
    parser.add_argument('-n', '--n_timesteps', type=int, default=20,
                        help='Number of timesteps.  Timesteps are nominally 6 months but in reality defined by the assumption that mean(p_update)=0.5')
    parser.add_argument('-r', '--randomize_beta', action='store_true',
                        help='Use randomized coefficients from H&L model rather than best estimates')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='generate plots rather than output results on command line')
    args = parser.parse_args()

    # Read in social graph (See ../NorthJutlandSocialGraph/README for file format)
    filename = '../NorthJutlandSocialGraph/'\
        f'NorthJutlandSocialGraph_{args.mean_n_weak_ties}_{args.modal_weak_tie_km}.pickle'
    coords, edges, strong_tie = pickle.loads(open(filename,'rb').read())
    M = coords.shape[0]

    # Run the model
    X,Y,X_vs_time,Y_vs_time = run_model(coords, edges, strong_tie,
                                        args.awareness_pc, args.facility_pc,
                                        args.p_update_logit_normal_sigma,
                                        args.n_timesteps, args.randomize_beta)

    # Output the results
    if args.interactive:
        shell = InteractiveShellEmbed()
        shell.enable_matplotlib()
        plt.ion()
        plot_vs_time(X_vs_time, Y_vs_time)
#        plot_scatter_diagram_on_map(coords,Y)
        plot_stages_of_change_distribution(coords,Y)
        shell()
    else:
        print(f'X={X_vs_time[-1].tolist()}\nY={Y_vs_time[-1].tolist()}')

