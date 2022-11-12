from typing import Tuple, Callable
import numpy as np
from numpy.typing import ArrayLike
from numba import njit
from chilife import jaccard, dirichlet

# TODO: Move int examples
@njit(cache=True)
def optimize_weights(
    ensemble: ArrayLike,
    idx: ArrayLike,
    start_weights: ArrayLike,
    start_score: float,
    data: ArrayLike,
    score_func: Callable = jaccard
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, float]:
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
