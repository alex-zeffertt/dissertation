#!/usr/bin/env python3
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
plt.ion()

# Parameters

# Social Graph parameters
mean_n_weak_ties  = 6   if len(sys.argv) <= 1 else int(sys.argv[1])
modal_weak_tie_km = 1.0 if len(sys.argv) <= 2 else float(sys.argv[2])
awareness_pc = 30       if len(sys.argv) <= 3 else float(sys.argv[3])
facility_pc = 33.333333 if len(sys.argv) <= 4 else float(sys.argv[4])
p_update_logit_normal_sigma = 0.5 if len(sys.argv) <= 5 else float(sys.argv[5])
n_timesteps = 20        if len(sys.argv) <= 6 else int(sys.argv[6])

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
    [0.0, 0.73, 2.54, 3.19, 3.30]])  # Reducer vs intention

# Calculate the NOintention:intention:reducer split for case X = 0 (no ties)
pop_not_intending_with_no_ties = 100 - awareness_pc
pop_reducing_with_no_ties = awareness_pc * facility_pc / 100
pop_intending_with_no_ties = 100 - pop_not_intending_with_no_ties - pop_reducing_with_no_ties

# Calculate P[:,:] as per calcProbs
percent = [pop_not_intending_with_no_ties, pop_intending_with_no_ties, pop_reducing_with_no_ties]
C = np.log(np.array([percent])/percent[0])
C = C.transpose()
P = np.exp(C + beta)/np.exp(C + beta).sum(axis=0)

# Read in the social graph selected.
# This is an undirected graph.
#
# coords rows are [lng_deg,lat_deg] for a single person
# edges  rows are [idx1,idx2]       where idx1>idx2 are indices into coords
# strong_tie[idx] is True or False  where idx is an index into edges

filename = '../NorthJutlandSocialGraph/'\
    f'NorthJutlandSocialGraph_{mean_n_weak_ties}_{modal_weak_tie_km}.pickle'
coords, edges, strong_tie = pickle.loads(open(filename,'rb').read())

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
Y = np.ndarray(M, dtype=int)

# Start off by assigning at random according to the probabilities for current state
for j in range(n_tie_categories):
    idxs = (X == j).nonzero()[0]
    n_idxs = len(idxs)
    probs = P[:,j]
    Y[idxs] = np.random.choice(n_diet_categories, n_idxs, p=probs)

# Different individuals show different resistance to change.
# We characterize this by assigning to each idx a probability
# that it's state Y[idx] will be updated in one time step.
# Thus if p_update[idx] = 0 then Y[idx] will never change, and if
# p_update[idx] = 0.5 then Y[idx] will be updated with a probability of 1/2
# in any given time step.  (Even if it is updated Y[idx] may be updated
# with the same value, since the value selected will be random with probabilities
# based on X[idx].)
#
# We assume that logit(p_update) is normally distributed with mean of 0
# (since changing the mean is equivalent to just altering the interpretation
# of the time step) and s.d. p_update_logit_normal_sigma.
def logit_normal_pdf(x,mu,sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*\
        (1/(x*(1-x)))*np.exp(-(np.log(x/(1-x)))**2/(2*sigma**2))

p_update_vals = np.linspace(0.00001,0.99999,1000)
p_update_probs = logit_normal_pdf(p_update_vals, 0, p_update_logit_normal_sigma)
p_update_probs /= p_update_probs.sum()
p_update = np.random.choice(p_update_vals, M, p=p_update_probs)

# TODO plot maps

# Arrays for saving changes in populations over time, for latter plotting
X_vs_time = np.zeros((n_timesteps,n_tie_categories))
Y_vs_time = np.zeros((n_timesteps,n_diet_categories))

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
    X[(n_strong_ties >= 2)] = 4

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
