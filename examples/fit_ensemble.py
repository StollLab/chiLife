import multiprocessing as mp
import argparse, warnings, pickle
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

import MDAnalysis as mda
import chilife as xl
from chilife.numba_utils import dirichlet



parser = argparse.ArgumentParser(description="fit_ensemble accepts a directory of pdb structures and a set of DEER "
                                             "distance distributions and attempts to find the smallest set and relative "
                                             "ratios of pdb structures in the pool that best fit the provided experimental "
                                             "distance distributions")

parser.add_argument("-r", dest="restraints", metavar="RESTRAINTS FILE", default=None,
                    help="pickle file containing a dictionary of restraitns where the keys are the site pairs and the values "
                         "are the distance axis and probability distribution")
parser.add_argument("-d", dest="directory", metavar="POOL DIRECTORY", default='./pool/',
                    help="directory containing pool of pdbs to fit")
parser.add_argument("-n", dest="iter_n", metavar="NUMBER OF ITERATIONS", type=int, default=150000,
                    help="number of iterations of MC search")
parser.add_argument("-o", dest="out_file", metavar="OUTPUT FILE", default='best_ensemble',
                    help="Name of output file with best ensemble and SSE")
parser.add_argument("--reps", dest="reps", metavar="NUMBER OF replicates", type=int, default=3,
                    help="number of times to repeat of MC from the beginning")
args = parser.parse_args()


@njit
def loss(P, Q):
    return np.sum(-np.log(P/Q))

@njit(cache=True)
def optimize_weights(
    ensemble,
    idx,
    start_weights,
    start_score,
    data,
    score_func = loss):
    """Fit weights to an ensemble of distance distributions optimizing the score with respect to user defined data.

    Parameters
    ----------
    ensemble : ArrayLike
        Array of distance distributions for each structure and each site pair in the ensemble
    idx : ArrayLike
        Indices of structures in parent matrix of structures
    start_weights : ArrayLike
        Initial values of weights
    start_score : float
        Score of ensemble to data with initial weight values
    data : ArrayLike
        Experimental data to fit simulation to
    score_func : Callable, optional
        Function to minimize to determine optimal fit. This function should accept two vectors, ``Ptest`` and ``Ptrue``
        , and return a float comparing the similarity of ``Ptest`` and ``Ptrue`` .

    Returns
    -------
    ensemble : np.ndarray
        Array of distance distributions for the optimized ensemble
    idx : np.ndarray
        Indices of distance distributions in the optimized ensemble
    best_weights : np.ndarray
        Array of optimized weights corresponding to the optimized ensemble.
    best_score : float
        The value of ``score_func`` for the optimized ensemble
    """
    best_score = start_score
    best_weights = start_weights.copy()

    count = 0
    while count < 100:

        # Assign new weights from dirichlet distribution of best weights
        new_weights = dirichlet(best_weights)
        new_score = score_func(np.dot(ensemble, new_weights), data)

        # Keep score if improved
        if new_score > best_score:
            best_score = new_score

            # Drop all structures present at less than 0.1%
            ensemble = ensemble[:, new_weights > 1e-3]
            idx = idx[new_weights > 1e-3]
            best_weights = new_weights[new_weights > 1e-3]

            # Rest count
            count = 0

        else:
            count += 1

    return ensemble, idx, best_weights, best_score

with open(args.restraints, 'rb') as f:
    rst = pickle.load(f)

# Load all pdb files into Memory
pool = [str(p) for p in (Path(args.directory).glob('*.pdb'))]
print('Loading all structure files into memory. This may take a moment...')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    U = mda.Universe(pool[0], *pool, in_memory=True)


selection = U.select_atoms('protein')
def get_all_SLs(site):
    return [xl.SpinLabel('I1M', site=site, protein=selection) for ts in tqdm(U.trajectory)]


print('Simulating all distance distributions. This will take even longer...')
sites = set(chain.from_iterable(rst))
SLs = {site : get_all_SLs(site) for site in sites}

# get distributions
Ps = [[xl.distance_distribution(SL1i, SL2i, rst[sites][0])
       for SL1i, SL2i in tqdm(zip(SLs[sites[0]], SLs[sites[1]]))]
        for sites in rst]

A = np.column_stack(Ps)
y = np.concatenate([rst[sites][1] for sites in rst])

for j in range(args.reps):
    best_score = 0
    best_weights = None
    best_idx = None
    sim_dd = None
    # iterate iter_n times
    for i in tqdm(args.iter_n):

        # Get P(r)s for random set of 10 structures
        idx = np.random.randint(0, len(A), 10)
        trial_ens = A[idx].T

        # Assign evenly distributed weights
        start_weights = np.ones(10) / 10

        # Calculate initial score
        start_score = loss(trial_ens.dot(start_weights), y)

        trial_ens, idx, trial_weights, trial_score  = optimize_weights(trial_ens, idx, start_weights, start_score, y)

        # Keep ensemble if improved
        if trial_score > best_score:
            print('new higth score!: ', trial_score)

            best_idx = idx.copy()
            best_score = float(trial_score)
            best_ens = trial_ens.copy()
            best_weights = trial_weights.copy()
            best_sim_dd = trial_ens.dot(trial_weights)

            print('Ensemble: ', [pool[tidx] for tidx in best_idx])
            print('Weights: ', best_weights)

    # Print output
    best_pdbs = [pool[tidx] for tidx in best_idx]

    print("Best ensemble: ", best_pdbs)
    print("Weights: ", best_weights)
    print("Score: ", best_score)

    # Save output
    fname = args.out_file +  '_{0:02d}.txt'.format(j)
    with open(fname, 'w') as file:
        file.write("Best ensemble: " + str(best_pdbs) + "\n")
        file.write("Weights: " + str(best_weights) + "\n")
        file.write("Score: " + str(best_score) + "\n")

    simulated_distributions = A[best_idx]
    np.save(args.out_file + '_{0:02d}.npy'.format(j), simulated_distributions)

    # Plot output
    fig, ax = plt.subplots()
    ax.plot(y, label='Experimental')
    ax.plot(best_sim_dd, label='Simulated')

    #plt.show()
    fig.savefig(args.out_file + '_{0:02d}.png'.format(j), dpi=300, bbox_inches='tight')
    plt.close()

