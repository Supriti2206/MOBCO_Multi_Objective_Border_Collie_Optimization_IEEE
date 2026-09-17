"""
normalization.py — Normalize fitness for crowding distance.
"""

import numpy as np

def normalize_fitness(fitness, objective_types):
    """
    Min-max normalize each objective column to [0, 1], flipping 'max'
    objectives so that after normalization, smaller is always better across
    every column (this is what lets crowding distance treat all objectives
    uniformly regardless of their original direction/scale).

    Parameters:
        fitness: (N, n_obj) raw fitness values
        objective_types: list like ['min', 'max', ...], one per column

    Returns:
        normalized: (N, n_obj) array in [0, 1] per column; a column with
                    (near) zero range collapses to a constant 0.5 for every
                    row, since there's nothing to distinguish within it.
    """
    fitness = np.asarray(fitness, dtype=float)
    normalized = fitness.copy()

    for j, obj_type in enumerate(objective_types):
        col = fitness[:, j]
        f_min, f_max = col.min(), col.max()

        if f_max - f_min > 1e-12:
            if obj_type == 'min':
                normalized[:, j] = (col - f_min) / (f_max - f_min)
            else:  # 'max'
                normalized[:, j] = (f_max - col) / (f_max - f_min)  # Invert so smaller = better
        else:
            # All values in this column are (numerically) identical -- no
            # basis for ranking, so give every row the same mid-point value
            normalized[:, j] = 0.5

    return normalized