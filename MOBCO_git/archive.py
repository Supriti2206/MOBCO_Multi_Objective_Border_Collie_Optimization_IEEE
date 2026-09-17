import numpy as np
from nondominated_sort import fast_nondominated_sort
from crowding_distance import crowding_distance_mixed


class Archive:
    """
    External (non-dominated) archive used to track the best solutions seen
    across all iterations, independent of the current working population.

    Every call to `update` merges the new candidates into the archive,
    re-filters down to the mutually non-dominated (rank-0) front, and -- if
    that front has grown beyond `max_size` -- prunes the most crowded points
    until it fits (see `_truncate`).
    """

    def __init__(self, objective_types, max_size=200):
        self.objective_types = objective_types  # e.g. ['min', 'min', 'max', ...]
        self.max_size = max_size
        self.decisions = np.empty((0, 0))  # decision vectors; shape fixed on first update()
        self.fitness = np.empty((0, len(objective_types)))
        self._initialized = False  # becomes True once `decisions` has its real column width

    def _dedupe(self, decisions, fitness):
        """Remove duplicate decision vectors (rounded to 10 decimals) so the
        same solution isn't kept multiple times in the archive."""
        if len(decisions) == 0:
            return decisions, fitness
        _, unique_idx = np.unique(np.round(decisions, 10), axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)  # preserve original relative ordering
        return decisions[unique_idx], fitness[unique_idx]

    def update(self, decisions, fitness):
        """Merge candidate (decisions, fitness) into the archive."""
        decisions = np.asarray(decisions)
        fitness = np.asarray(fitness)

        if not self._initialized:
            self.decisions = np.empty((0, decisions.shape[1]))
            self._initialized = True

        combined_dec = np.vstack([self.decisions, decisions]) if len(self.decisions) else decisions
        combined_fit = np.vstack([self.fitness, fitness]) if len(self.fitness) else fitness

        combined_dec, combined_fit = self._dedupe(combined_dec, combined_fit)

        # Keep only mutually non-dominated solutions (rank-0 front)
        fronts = fast_nondominated_sort(combined_fit, self.objective_types)
        keep = fronts[0] if fronts else list(range(len(combined_fit)))

        self.decisions = combined_dec[keep]
        self.fitness = combined_fit[keep]

        if len(self.fitness) > self.max_size:
            self._truncate()

    def _truncate(self):
        """Reduce archive to max_size, discarding the most crowded points."""
        n = len(self.fitness)
        indices = list(range(n))
        while len(indices) > self.max_size:
            distances = crowding_distance_mixed(self.fitness, indices, self.objective_types)
            # remove the single most-crowded (smallest distance) solution
            worst_pos = int(np.argmin(distances))
            indices.pop(worst_pos)
        self.decisions = self.decisions[indices]
        self.fitness = self.fitness[indices]

    def get(self):
        return self.decisions, self.fitness

    def __len__(self):
        return len(self.fitness)
