import numpy as np


def mobco_check(pop, n, L, ub, lb, acc, Vt, t, rng=None):
    """
    Enforce search-space bounds and numerical sanity after the position
    update + mutation steps, then re-randomize anything that went bad.

    Rather than clamping out-of-bounds values back onto the boundary, this
    re-initializes the offending position/acceleration/velocity cell(s) (and,
    if the per-agent time step is non-finite, that agent's entire row) with
    fresh random values -- effectively "resetting" broken agents instead of
    projecting them back into the feasible region.

    Parameters:
        pop: (n, L) current positions
        n: population size
        L: number of decision variables
        ub, lb: upper/lower bounds, scalar or length-L, broadcast to (L,)
        acc: (n, L) accelerations
        Vt: (n, L) velocities
        t: (n,) per-agent time step
        rng: numpy Generator to use for randomness (falls back to the global
             np.random state if not provided)

    Returns:
        (pop1, acc1, t1, Vt1): the same arrays with bad cells/rows replaced.
    """
    r = rng if rng is not None else np.random

    ub = np.broadcast_to(np.asarray(ub, dtype=float), (L,))
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (L,))

    pop1 = pop.copy()
    acc1 = acc.copy()
    t1 = t.copy()
    Vt1 = Vt.copy()

    # An agent-dimension is "bad" if its position left the box, or if
    # any of its kinematic state (position/acc/velocity/time) is
    # non-finite (NaN or +/-Inf).
    out_of_bounds = (pop < lb[None, :]) | (pop > ub[None, :])
    bad_acc = ~np.isfinite(acc)
    bad_vt = ~np.isfinite(Vt)
    bad_pos = ~np.isfinite(pop)
    bad_cell = out_of_bounds | bad_acc | bad_vt | bad_pos

    # An agent whose time-step became non-finite gets its whole row
    # (every dimension) re-initialised.
    bad_row = ~np.isfinite(t1)
    bad_cell |= bad_row[:, None]

    n_bad = int(np.count_nonzero(bad_cell))
    if n_bad:
        fresh_pos = lb[None, :] + r.random((n, L)) * (ub - lb)[None, :]
        fresh_acc = r.random((n, L))
        fresh_vt = r.standard_normal((n, L)) * 0.5
        pop1 = np.where(bad_cell, fresh_pos, pop1)
        acc1 = np.where(bad_cell, fresh_acc, acc1)
        Vt1 = np.where(bad_cell, fresh_vt, Vt1)

    if np.any(bad_row):
        fresh_t = r.random(n)
        fresh_t[fresh_t == 0] = 1e-6
        t1 = np.where(bad_row, fresh_t, t1)

    return pop1, acc1, t1, Vt1
