import numpy as np

def mobco_fitness(population, n, dim, fobj, n_obj, objective_types=None):
    """
    Evaluate every individual in the population on the objective function.

    Parameters:
        population: (n, dim) array of candidate solutions (decision vectors)
        n: number of individuals in the population
        dim: number of decision variables per individual (unused directly here,
             kept for a consistent call signature with the rest of the codebase)
        fobj: the objective function; called as fobj(individual) and expected to
              return a length-n_obj array/list of objective values
        n_obj: number of objectives
        objective_types: list like ['min', 'max', ...] of length n_obj; currently
                          only used to default to all-minimization, no
                          min/max conversion is applied here (that happens
                          elsewhere, e.g. in dominance comparisons)

    Returns:
        fitness: (n, n_obj) array where row i holds the objective values for
                  population[i]
    """
    # If no types specified, all are minimization
    if objective_types is None:
        objective_types = ['min'] * n_obj

    fitness = np.zeros((n, n_obj))

    # Evaluate each individual one at a time (no vectorized batch evaluation)
    for i in range(n):
        fitness[i, :] = fobj(population[i, :])

    return fitness