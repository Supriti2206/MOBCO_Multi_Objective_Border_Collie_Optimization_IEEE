"""
metrics.py -- Performance indicators used to compare algorithms/problems in
the IEEE paper: Hypervolume (HV), Inverted Generational Distance (IGD),
Generational Distance (GD), Spacing, generalized Spread (Delta), and the
additive Epsilon indicator.

All indicators are computed relative to a `MetricContext`, which fixes a
*shared* reference front, reference point, and scaling for one problem so
that HV/IGD/etc. are comparable across different algorithms and different
runs of the same problem (rather than each run/algorithm being scored
against its own ad hoc bounding box).
"""

import numpy as np

from dominance import min_mask, nondominated_mask


# Context
class MetricContext:
    """
    Precomputes everything needed to score obtained fronts against one
    problem's analytic reference front, ONCE per problem, so every
    algorithm/run/iteration can reuse identical scaling -- this is what makes
    HV and the other indicators comparable across runs and across
    algorithms for the same problem.

    Parameters:
        ref_front: (R, n_obj) array of reference (true Pareto front) points
        objective_types: list like ['min', 'max', ...]
        margin: fraction of the reference front's range used to push the
                hypervolume reference point out beyond the worst (nadir-side)
                corner of the reference front, so obtained points near the
                nadir still contribute some hypervolume
        seed: seed for this context's own RNG (used by the Monte-Carlo HV
              estimator, so repeated evaluate_front calls using this context
              are reproducible)
    """

    def __init__(self, ref_front, objective_types, margin=0.10, seed=12345):
        self.objective_types = list(objective_types)
        self.mins = min_mask(objective_types)
        self.R = np.asarray(ref_front, dtype=float)
        self.margin = margin
        self.rng = np.random.default_rng(seed)

        if self.R.size == 0:
            raise ValueError("empty reference front")

        self.lo = self.R.min(axis=0)
        self.hi = self.R.max(axis=0)
        span = self.hi - self.lo
        span[span <= 0] = 1.0   # guard against a degenerate (zero-range) objective
        self.span = span

        # direction-aware best / worst corners of the reference front
        self.best = np.where(self.mins, self.lo, self.hi)
        self.worst = np.where(self.mins, self.hi, self.lo)
        # reference (nadir-side) point, pushed out by `margin` of the range
        self.ref_point = np.where(self.mins,
                                  self.worst + margin * self.span,
                                  self.worst - margin * self.span)
        self.hv_span = np.abs(self.ref_point - self.best)
        self.hv_span[self.hv_span <= 0] = 1.0

        self.Rz = self.scale(self.R)
        self.R_extremes = self._extremes()

    # scaling helpers 
    def scale(self, F):
        """Min-max scale into the reference-front box (no direction change)."""
        F = np.asarray(F, dtype=float)
        if F.size == 0:
            return F.reshape(0, len(self.objective_types))
        return (F - self.lo) / self.span

    def utility(self, F):
        """
        Axis-wise distance towards the reference point, normalised to [0, 1].
        u_j = 1 means 'as good as the best reference value on objective j',
        u_j <= 0 means 'no better than the reference point'.
        Used only by the hypervolume integral.
        """
        F = np.asarray(F, dtype=float)
        if F.size == 0:
            return F.reshape(0, len(self.objective_types))
        signed = np.where(self.mins, self.ref_point - F, F - self.ref_point)
        return np.clip(signed / self.hv_span, 0.0, 1.0)

    def _extremes(self):
        """Index of the best reference point on each objective."""
        idx = []
        for j, is_min in enumerate(self.mins):
            idx.append(int(np.argmin(self.R[:, j]) if is_min
                           else np.argmax(self.R[:, j])))
        return self.Rz[np.array(idx)]


# Hypervolume
def _hv_2d_exact(U):
    """Exact HV of the union of boxes [0,u] in 2-D (U already in [0,1]^2).

    Works by sorting points by decreasing x, then sweeping and only adding
    the "new" strip of area not already covered by a previously-seen (larger)
    y value -- the standard exact 2D hypervolume sweep algorithm."""
    U = U[np.all(U > 0, axis=1)]   # drop points with zero utility on any axis (contribute nothing)
    if len(U) == 0:
        return 0.0
    order = np.lexsort((-U[:, 1], -U[:, 0]))  # sort by x desc, tie-break by y desc
    U = U[order]
    hv, prev_y = 0.0, 0.0
    for x, y in U:
        if y > prev_y:
            hv += x * (y - prev_y)  # new horizontal strip of height (y - prev_y), width x
            prev_y = y
    return float(hv)


def _hv_mc(U, n_samples, rng, chunk=4000):
    """Monte-Carlo HV of the union of boxes [0,u] inside the unit cube.

    Used for 3+ objectives, where the exact sweep algorithm above doesn't
    apply. Draws random points in the unit cube and estimates HV as the
    fraction "covered" (dominated by at least one point in U), processed in
    chunks to bound peak memory for the (chunk x len(U) x m) comparison
    tensor."""
    U = U[np.all(U > 0, axis=1)]
    if len(U) == 0:
        return 0.0
    m = U.shape[1]
    hits = 0
    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        S = rng.random((b, m))
        # sample is covered if some u dominates it component-wise
        covered = np.any(np.all(U[None, :, :] >= S[:, None, :], axis=2), axis=1)
        hits += int(covered.sum())
        done += b
    return hits / float(n_samples)


def hypervolume(F, ctx, n_samples=None):
    """
    Normalised hypervolume in [0, 1] (1.0 == the whole reference box is
    dominated).  Exact for 2 objectives, Monte-Carlo for 3+.
    """
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return float('nan')
    F = F[np.all(np.isfinite(F), axis=1)]
    if len(F) == 0:
        return float('nan')
    U = ctx.utility(F)  # map raw objectives -> [0,1] utility relative to ctx.ref_point
    # Keep only non-dominated utility points -- dominated points don't add
    # any hypervolume, so dropping them speeds up the exact/MC computation
    U = U[nondominated_mask(U, ['max'] * U.shape[1])]
    if U.shape[1] == 2:
        return _hv_2d_exact(U)
    n_samples = 20000 if n_samples is None else n_samples
    return _hv_mc(U, n_samples, ctx.rng)


# Distance-based indicators
def _pairwise(A, B):
    """Euclidean distance matrix between every row of A and every row of B."""
    d = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.einsum('ijk,ijk->ij', d, d))


def generational_distance(F, ctx, p=1):
    """GD = mean distance from each obtained point to the nearest reference point."""
    A = ctx.scale(F)
    if len(A) == 0:
        return float('nan')
    d = _pairwise(A, ctx.Rz).min(axis=1)
    return float(np.mean(d ** p) ** (1.0 / p))


def inverted_generational_distance(F, ctx, p=1):
    """IGD = mean distance from each reference point to the nearest obtained point."""
    A = ctx.scale(F)
    if len(A) == 0:
        return float('nan')
    d = _pairwise(ctx.Rz, A).min(axis=1)
    return float(np.mean(d ** p) ** (1.0 / p))


def spacing(F, ctx):
    """Standard deviation of the nearest-neighbour distances inside the front."""
    A = ctx.scale(F)
    n = len(A)
    if n < 2:
        return float('nan')
    d = _pairwise(A, A)
    np.fill_diagonal(d, np.inf)
    dmin = d.min(axis=1)
    return float(np.sqrt(np.sum((dmin.mean() - dmin) ** 2) / (n - 1)))


def spread(F, ctx):
    """
    Generalised spread (Delta).  Uses the extreme points of the reference
    front, so it works for 2, 3 and 4 objectives alike.  Lower is better.
    """
    A = ctx.scale(F)
    n = len(A)
    if n < 2:
        return float('nan')
    d = _pairwise(A, A)
    np.fill_diagonal(d, np.inf)
    dmin = d.min(axis=1)
    dbar = dmin.mean()
    d_extreme = _pairwise(ctx.R_extremes, A).min(axis=1).sum()
    denom = d_extreme + n * dbar
    if denom <= 0:
        return float('nan')
    return float((d_extreme + np.sum(np.abs(dmin - dbar))) / denom)


def epsilon_indicator(F, ctx):
    """
    Additive epsilon indicator I_eps+(A, R): the smallest amount by which every
    reference point must be relaxed for the obtained set to cover it.
    Directions are applied here (a maximised objective is relaxed downwards).

    Computed as max over reference points of (min over obtained points of
    (max over objectives of the signed per-objective gap)) -- the classic
    min-max-max formulation of the additive epsilon indicator.
    """
    A = ctx.scale(F)
    if len(A) == 0:
        return float('nan')
    sign = np.where(ctx.mins, 1.0, -1.0)
    diff = sign[None, None, :] * (A[None, :, :] - ctx.Rz[:, None, :])
    return float(np.max(np.min(np.max(diff, axis=2), axis=1)))


# Bundle
def evaluate_front(F, ctx, hv_samples=None):
    """Compute all six indicators (HV, IGD, GD, Spacing, Spread, Epsilon) plus
    the front's size, for one obtained front `F` against the given
    MetricContext. Returns a dict of NaNs (with size=0) if F is empty."""
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        nan = float('nan')
        return {'hv': nan, 'igd': nan, 'gd': nan, 'spacing': nan,
                'spread': nan, 'epsilon': nan, 'size': 0}
    return {
        'hv': hypervolume(F, ctx, hv_samples),
        'igd': inverted_generational_distance(F, ctx),
        'gd': generational_distance(F, ctx),
        'spacing': spacing(F, ctx),
        'spread': spread(F, ctx),
        'epsilon': epsilon_indicator(F, ctx),
        'size': int(len(F)),
    }


METRIC_DIRECTION = {          # True = larger is better; used e.g. by
                              # visualization/excel code to know which
                              # direction "improvement" means per metric
    'Hypervolume': True, 'IGD': False, 'GD': False,
    'Spacing': False, 'Spread': False, 'Epsilon': False, 'Runtime': False,
}
