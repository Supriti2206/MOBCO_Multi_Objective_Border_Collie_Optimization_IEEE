import numpy as np
from normalization import normalize_fitness  # <-- ADD THIS

def crowding_distance_mixed(fitness_values, front_indices, objective_types):
    """
    Calculate the crowding distance for the solutions listed
    in `front_indices`, using min-max NORMALIZED fitness values so objectives
    on different scales (and 'min' vs 'max' directions) contribute comparably.

    Parameters:
        fitness_values: (N, n_obj) full fitness array (front_indices index
                         into this)
        front_indices: list/array of row indices into fitness_values that
                        belong to the front being measured
        objective_types: list like ['min', 'max', ...], one per objective;
                          used by normalize_fitness to invert 'max' objectives
                          onto the same [0, 1] "smaller is better" scale

    Returns:
        distances: array aligned POSITIONALLY with `front_indices` (i.e.
                    distances[i] is the crowding distance of front_indices[i]),
                    NOT indexed by the raw fitness row index.

    Larger distance = more isolated / less crowded (kept preferentially when
    the archive needs to be pruned in Archive._truncate).
    """
    n_front = len(front_indices)
    distances = np.zeros(n_front)

    # With 2 or fewer points there's no meaningful "crowding" -- treat both
    # (all) as maximally isolated so neither is preferentially discarded.
    if n_front <= 2:
        distances[:] = np.inf
        return distances

    # Normalize first
    normalized_fit = normalize_fitness(fitness_values, objective_types)

    n_obj = len(objective_types)

    for obj in range(n_obj):
        # Sort the front's *indices* by this objective's normalized value
        # (already in [0,1] with 'max' objectives inverted so smaller-is-better)
        sorted_indices = sorted(front_indices, key=lambda idx: normalized_fit[idx, obj])

        # NOTE: this marks distances[0] and distances[-1] -- i.e. the first
        # and last entries of `front_indices` in their ORIGINAL (unsorted)
        # order -- as infinitely crowded-distance (boundary points), rather
        # than the actual extreme points of `sorted_indices` for THIS
        # objective. In standard NSGA-II crowding distance, the two boundary
        # solutions for *each* objective's sort should get +inf. As written,
        # only the first/last positions of the original front_indices list
        # get +inf, repeated (redundantly) on every objective's loop
        # iteration. Worth double-checking against the intended algorithm --
        # this looks like it may not do what the comment/name implies.
        distances[0] = np.inf
        distances[-1] = np.inf

        f_max = normalized_fit[sorted_indices[-1], obj]
        f_min = normalized_fit[sorted_indices[0], obj]

        if f_max != f_min:
            # Interior points: distance += normalized gap between neighbors,
            # scaled by this objective's normalized range
            for i in range(1, n_front - 1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]
                dist = (normalized_fit[next_idx, obj] - normalized_fit[prev_idx, obj]) / (f_max - f_min)
                distances[i] += dist  # accumulated across all objectives

    # Handle any numerical issues (e.g. inf - inf producing nan)
    distances = np.nan_to_num(distances, nan=0.0, posinf=1e10)
    return distances


def sort_by_crowding_distance_mixed(population, fitness, front_indices, objective_types):
    """
    Sort solutions in a front by crowding distance (descending)
    """
    distances = crowding_distance_mixed(fitness, front_indices, objective_types)
    sorted_indices = sorted(front_indices, key=lambda idx: distances[idx], reverse=True)
    return sorted_indices, distances