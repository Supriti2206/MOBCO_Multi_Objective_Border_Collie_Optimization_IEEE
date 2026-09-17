import numpy as np
from nondominated_sort import fast_nondominated_sort
from crowding_distance import crowding_distance_mixed
from normalization import normalize_fitness

def mobco_herding(pop, Vt, fit, n, L, acc, t, objective_types):
    """
    "Herding" step of MOBCO: reorder the whole population (positions,
    velocities, fitness, accelerations, time-steps) so that the best
    individuals come first. Ordering is by (Pareto rank ascending, crowding
    distance descending) -- i.e. prefer lower (better) rank, and within a 
    rank prefer more isolated (less crowded) solutions.

    This ordering is what optimizer.py relies on to identify the current
    "lead dogs" (the first n_dogs individuals after this sort) each
    iteration.

    Parameters:
        pop, Vt, fit, acc, t: parallel arrays describing the population's
            positions, velocities, fitness, accelerations, and per-agent
            time-step (all indexed by the same row order coming in)
        n: population size
        L: number of decision variables (unused directly, kept for a
           consistent call signature)
        objective_types: list like ['min', 'max', ...]

    Returns:
        (sorted_pop, sorted_Vt, sorted_fit, sorted_acc, sorted_t): the same
        arrays, all reordered by the same permutation (best-first).
    """
    # NOTE: this normalized_fit is computed but never used below --
    # crowding_distance_mixed() re-normalizes internally on its own (see
    # normalization.py), so this line is redundant dead computation.
    normalized_fit = normalize_fitness(fit, objective_types)

    # Perform non-dominated sorting (uses raw fitness for dominance)
    fronts = fast_nondominated_sort(fit, objective_types)

    # Calculate crowding distance per front, using NORMALIZED fitness
    # (normalization happens inside crowding_distance_mixed itself)
    all_distances = np.zeros(n)
    for front in fronts:
        distances = crowding_distance_mixed(fit, front, objective_types)
        for i, idx in enumerate(front):
            all_distances[idx] = distances[i]

    # Assign each individual its front/rank number (0 = best front)
    ranks = np.zeros(n)
    for rank, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank

    # Sort indices by rank first (ascending, better rank first), then by
    # crowding distance (descending, more isolated first) as the tiebreaker
    sorted_indices = sorted(range(n), key=lambda i: (ranks[i], -all_distances[i]))

    # Reorder all parallel arrays using the same permutation
    sorted_pop = pop[sorted_indices, :]
    sorted_Vt = Vt[sorted_indices, :]
    sorted_fit = fit[sorted_indices, :]
    sorted_acc = acc[sorted_indices, :]
    sorted_t = t[sorted_indices]

    return sorted_pop, sorted_Vt, sorted_fit, sorted_acc, sorted_t

