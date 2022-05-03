import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# TODO make into object so you can run one timestep at a time
# and set up initial states
def run_model(n_rows, n_cols,
              awareness_pc, facility_pc, # Global parameters affecting CA model
              n_timesteps,               # How much simulated time to run
              return_history=False):

    # blah
    n_tie_categories = 5
    n_diet_categories = 3

    # From H&L paper
    beta = np.array([
        [0.0, 0.0,  0.0,  0.0,  0.0],
        [0.0, 0.69, 2.78, 3.81, 0.65],   # Intention vs NO intention
        [0.0, 0.73, 2.54, 3.19, 3.30]])  # Reducer vs intention

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

    # Create grid on which CA works
    x_grid, y_grid = np.mgrid[0:1:n_rows*1j, 0:1:n_cols*1j]

    # X is explanatory variable (ties to reducers category) and
    # Y is response variable (stage of change category)
    M = n_rows*n_cols
    X = np.zeros(M, dtype=int)
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

    def update_X():
        # Update tie categories X
        n_strong_ties = (Y[peers_strong] == 2).sum(axis=1)
        n_weak_ties   = (Y[peers_weak] == 2).sum(axis=1)
        X[(n_strong_ties == 0) & (n_weak_ties == 0)] = 0
        X[(n_strong_ties == 0) & (n_weak_ties >= 1)] = 1
        X[(n_strong_ties >= 1) & (n_strong_ties <= 2) & (n_weak_ties == 0)] = 2
        X[(n_strong_ties >= 1) & (n_strong_ties <= 2) & (n_weak_ties >= 1)] = 3
        X[(n_strong_ties >= 3)] = 4

    # initial state
    # TODO move this to argument
    Y[idxs[45:55][:,45:55].flatten()] = 2
    update_X()

    if return_history:
        results = [(np.reshape(X.copy(),(n_rows,n_cols)),
                    np.reshape(Y.copy(),(n_rows,n_cols)))]
        
    # The main loop:
    # every time step update X and then update Y then update plot
    for tstep in range(n_timesteps):
    
        # Update diet categories Y
        for j in range(n_tie_categories):
            # Find indices of individuals with this category of ties to reducers
            idxs = (X == j).nonzero()[0]

            # Change diet randomly using probs for social n/w categegory j
            Y[idxs] = np.random.choice(n_diet_categories, len(idxs), p=P[:,j])

        update_X()

        if return_history:
            results.append((
                np.reshape(X.copy(),(n_rows,n_cols)),
                np.reshape(Y.copy(),(n_rows,n_cols))))
        
    # Fix up shapes
    X.shape = n_rows,n_cols
    Y.shape = n_rows,n_cols

    return results if return_history else (X,Y)

if __name__ == '__main__':

    plt.ion()

    # Q: How quickly does it just become noise?
    # A: About 3
    
    fig, axes = plt.subplots(nrows=2,ncols=4, gridspec_kw={'width_ratios':[1,1,1,1.065]})
    plt.suptitle('Simple CA model over 3 generations\n\n')

    results = run_model(n_rows=100,n_cols=100,
                        awareness_pc=30, facility_pc=33.3,
                        n_timesteps=3,
                        return_history=True)
    
    for n_timesteps in range(4):

        X = results[n_timesteps][0]
        Y = results[n_timesteps][1]
        
        cmap = matplotlib.cm.get_cmap("coolwarm", 3)
        plt.sca(axes[0][n_timesteps])
        im = plt.imshow(Y, cmap=cmap, interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{n_timesteps} steps")
        if n_timesteps == 3:
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax, ticks=[.25,1,1.75])
            cb.ax.set_yticklabels(['   NO\nintention','Intention','Reducer'])
            
        cmap = matplotlib.cm.get_cmap("coolwarm", 5)
        plt.sca(axes[1][n_timesteps])
        im = plt.imshow(X, cmap=cmap, interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        if n_timesteps == 3:
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax, ticks=[.25,1,2,3,3.75])
            cb.ax.set_yticklabels(['no ties','weak ties',
                                   '1-2 strong\nno weak','1-2 strong\n+weak',
                                   '3+ strong'])
        
