"""
    population : (n, L)  -- decision-variable positions, uniform in [lb, ub]
    Vt         : (n, L)  -- initial velocity of each agent
    acc        : (n, L)  -- initial acceleration of each agent
    t          : (n,)    -- initial "time step" scalar per agent

"""

import numpy as np


def mobco_generate(n, L, ub, lb, rng=None):
    """
    Randomly initialize a population and its kinematic state for MOBCO.

    Parameters:
        n: population size
        L: number of decision variables
        ub, lb: upper/lower bounds (broadcastable to shape (L,))
        rng: numpy Generator for reproducibility (falls back to global
             np.random if not given)

    Returns: (population, Vt, acc, t) -- see module docstring above for shapes.
    """
    ub = np.asarray(ub, dtype=float)
    lb = np.asarray(lb, dtype=float)
    r = rng if rng is not None else np.random

    population = lb + r.random((n, L)) * (ub - lb)

    # Small initial velocities/accelerations: agents start close to rest and
    # build up motion once the herding/leader signal kicks in.
    Vt = r.standard_normal((n, L)) * 0.5
    acc = r.random((n, L))

    # Per-agent "time step" used in s = u*t + 0.5*a*t^2. Kept in (0, 1] so a
    # single generation never moves an agent further than one bound-width.
    t = r.random(n)
    t[t == 0] = 1e-6

    return population, Vt, acc, t
