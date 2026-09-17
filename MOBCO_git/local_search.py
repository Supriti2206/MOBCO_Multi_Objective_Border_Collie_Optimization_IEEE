"""
local_search.py -- A local ("hill-climbing"/random-step) refinement pass
intended to nudge a handful of archive members towards dominance
improvements after MOBCO's main loop finishes.

*** NOT CURRENTLY USED ***
`refine_archive` (and its helper `_select_refinement_targets`) are not
imported or called anywhere else in this codebase (checked: no references in
optimizer.py, run_experiment.py, main.py, or elsewhere). This appears to be
either an abandoned feature or a hook intended for future use -- keeping it
here for reference/reuse, but be aware it has no effect on the current
experiment pipeline unless something is wired up to call it.
"""

import numpy as np

from dominance import dominates
from crowding_distance import crowding_distance_mixed


def _select_refinement_targets(archive, k):
    """Pick up to k archive members to refine: always protect each
    objective's single best ("extreme") point, then fill the rest of the
    quota with the most isolated (highest crowding-distance) members --
    i.e. prioritize spreading out sparse regions of the front rather than
    refining already-dense clusters."""
    n = len(archive)
    if n == 0:
        return []
    idx_all = list(range(n))
    distances = crowding_distance_mixed(archive.fitness, idx_all, archive.objective_types)

    protect = set()
    for j, t in enumerate(archive.objective_types):
        col = archive.fitness[:, j]
        protect.add(int(col.argmin()) if t == 'min' else int(col.argmax()))

    ranked = sorted(idx_all, key=lambda i: -distances[i])
    targets = list(protect)
    for i in ranked:
        if len(targets) >= k:
            break
        if i not in targets:
            targets.append(i)
    return targets[:k]


def refine_archive(archive, problem, lower, upper, rng,
                    k=3, n_tries=12, step_frac=0.05, shrink=0.6):
    """
    Attempt a simple random-direction local search around `k` selected
    archive members, accepting a step only if it dominates the current point
    (i.e. a strict local improvement), with a shrinking step size on failed
    attempts (classic pattern-search style refinement).

    Parameters:
        archive: the Archive instance to refine (mutated in place)
        problem: problem dict (only 'evaluate' is used)
        lower, upper: search-space bounds (used to scale the initial step size)
        rng: numpy Generator for reproducibility
        k: number of archive members to target (see _select_refinement_targets)
        n_tries: max local-search attempts per target
        step_frac: initial step size as a fraction of the search-space span
        shrink: multiplicative step shrink factor after a failed attempt

    Returns:
        n_improved: total count of accepted (dominating) improvement steps
                    across all targets.
    """
    if len(archive) == 0:
        return 0

    evaluate = problem['evaluate']
    types = archive.objective_types
    span = upper - lower
    n_improved = 0

    targets = _select_refinement_targets(archive, k)
    for idx in targets:
        x = archive.decisions[idx].copy()
        f = archive.fitness[idx].copy()
        step = step_frac * span

        for _ in range(n_tries):
            direction = rng.standard_normal(x.shape)  # random search direction
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction /= norm
            x_new = np.clip(x + direction * step, lower, upper)
            f_new = np.asarray(evaluate(x_new))

            if dominates(f_new, f, types):
                x, f = x_new, f_new
                n_improved += 1
                # keep the step size on a small refinement scale; don't
                # grow it back up after a success
            else:
                step *= shrink  # failed step: shrink and try again (pattern search)
                if np.all(step < 1e-10 * span.clip(min=1e-12)):
                    break  # step size has collapsed to numerical noise -- give up on this target

        archive.decisions[idx] = x
        archive.fitness[idx] = f

    # a refined point may now dominate other archive members that weren't
    # touched -- re-filter to keep the archive a valid non-dominated set
    archive.update(archive.decisions, archive.fitness)
    return n_improved
