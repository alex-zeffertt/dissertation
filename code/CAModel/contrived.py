import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# probabilities must be constrained to [0,1] so we use the a transform
# e.g.:
#
#    Pr(0 to 1 change | n strong ties to reducers) =
#
#             1-exp(-np)
#
#    Note that at n=0, dPr/dn = p as required

def run_model(n_rows, n_cols,
              p,q,r,
              weak_strong_ratio,
              n_timesteps,
              return_history=False,
              history_stride=1):

    # Create grid on which CA works
    x_grid, y_grid = np.mgrid[0:1:n_rows*1j, 0:1:n_cols*1j]

    # Y is the response variable i.e. the stage of change category
    # 0 : NO intention (to reduce)
    # 1 : Intention
    # 2 : Reducer
    M = n_rows*n_cols
    Y = np.zeros(M, dtype=int)

    # Build an array showing which peers are strong and which are weak ties
    peers_strong = np.zeros((M,4),dtype=int)
    peers_weak   = np.zeros((M,4),dtype=int)
    idxs = np.arange(M, dtype=int)
    row_idxs = np.arange(n_rows, dtype=int)
    col_idxs = np.arange(n_cols, dtype=int)
    idxs.shape = (n_rows, n_cols)
    peers_strong[:,0] = idxs[:,(col_idxs + 1) % n_cols].flatten()
    peers_strong[:,1] = idxs[:,(col_idxs - 1) % n_cols].flatten()
    peers_strong[:,2] = idxs[(row_idxs + 1) % n_rows,:].flatten()
    peers_strong[:,3] = idxs[(row_idxs - 1) % n_rows,:].flatten()
    peers_weak[:,0] = idxs[(row_idxs-1)%n_rows][:,(col_idxs-1)%n_cols].flatten()
    peers_weak[:,1] = idxs[(row_idxs-1)%n_rows][:,(col_idxs+1)%n_cols].flatten()
    peers_weak[:,2] = idxs[(row_idxs+1)%n_rows][:,(col_idxs-1)%n_cols].flatten()
    peers_weak[:,3] = idxs[(row_idxs+1)%n_rows][:,(col_idxs+1)%n_cols].flatten()

    Y[idxs[35:45][:,35:45].flatten()] = 2
    Y[idxs[55:65][:,55:65].flatten()] = 2
    if return_history:
        results = [np.reshape(Y.copy(),(n_rows,n_cols))]

    # The main loop:
    # every time step update X and then update Y then update plot
    for tstep in range(n_timesteps):

        n_reducer_strong_ties = (Y[peers_strong] == 2).sum(axis=1)
        n_reducer_weak_ties   = (Y[peers_weak] == 2).sum(axis=1)
        n_reducer_effective_strong = \
            n_reducer_strong_ties + \
            n_reducer_weak_ties * weak_strong_ratio

        n_non_reducer_strong_ties = (Y[peers_strong] < 2).sum(axis=1)
        n_non_reducer_weak_ties   = (Y[peers_weak] < 2).sum(axis=1)
        n_non_reducer_effective_strong = \
            n_non_reducer_strong_ties + \
            n_non_reducer_weak_ties * weak_strong_ratio
        
        # P, Q, and R are the probabilities for transition 0->1, 1->2, and 2->0
        # respectively, taking into account number of peers
        P = 1 - np.exp(-n_reducer_effective_strong * p)
        Q = 1 - np.exp(-n_reducer_effective_strong * q)
        R = 1 - np.exp(-n_non_reducer_effective_strong * r)
    
        # Update diet categories Y

        # Changes from 0 to 1 (NO intention to Intention)
        idxs = (Y == 0).nonzero()[0]
        do_change = np.random.binomial(1,P[idxs])
        idxs = idxs[do_change == 1]
        Y[idxs] = 1

        # Changes from 1 to 2 (Intention to Reducer)
        idxs = (Y == 1).nonzero()[0]
        do_change = np.random.binomial(1,Q[idxs])
        idxs = idxs[do_change == 1]
        Y[idxs] = 2

        # Changes from 2 to 0 (relapse: Reducer to NO intention)
        idxs = (Y == 2).nonzero()[0]
        do_change = np.random.binomial(1,R[idxs])
        idxs = idxs[do_change == 1]
        Y[idxs] = 0
        
        if return_history and (tstep + 1) % history_stride == 0:
            results.append(np.reshape(Y.copy(),(n_rows,n_cols)))

    # return the stuff
    Y.shape = (n_rows, n_cols)
    return results if return_history else Y


if __name__ == '__main__':

    plt.ion()

    # Q: How quickly does it just become noise?
    fig, axes = plt.subplots(nrows=2,ncols=4, gridspec_kw={'width_ratios':[1,1,1,1.065]})
    plt.suptitle('Contrived CA model over 700 generations\n\n')

    results = run_model(n_rows=100,n_cols=100,
                        p=0.20, q=0.05, r=0.035, weak_strong_ratio=0.25,
                        n_timesteps=800,
                        return_history=True, history_stride=100)
    
    for i in range(8):

        n_timesteps = 100*i
        Y = results[i]
        
        cmap = matplotlib.cm.get_cmap("coolwarm", 3)
        plt.sca(axes[i//4][i%4])
        im = plt.imshow(Y, cmap=cmap, interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{n_timesteps} steps")
        if i%4 == 3:
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax, ticks=[.25,1,1.75])
            cb.ax.set_yticklabels(['   NO\nintention','Intention','Reducer'])
