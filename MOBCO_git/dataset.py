"""
dataset.py -- Central registry/factory for all 27 benchmark problems used in
the experiment. Every problem is represented as a plain dict with a common
shape (name, dim, n_obj, bounds, objective_types, evaluate, true_front,
objective_names) so the rest of the pipeline (optimizer, metrics,
visualization, excel_writer) can treat every problem uniformly regardless of
its source module.

27 problems

    12  EvoPINN physics-informed problems   (4 objectives, all 'min')
     5  ZDT1, ZDT2, ZDT3, ZDT4, ZDT6        (2 objectives, all 'min')
     7  DTLZ1 ... DTLZ7                     (3 objectives, all 'min')
     3  Mixed_ZDT1, Mixed_DTLZ1, Mixed_DTLZ2 (mixed 'min'/'max')

"""

import numpy as np

from standard_test_functions import STANDARD_BUILDERS
from mixed_test_functions import MIXED_BUILDERS
from evopinn_problem import get_all_evopinn_problems


# EvoPINN adapters
def _wrap_evopinn(key, inst):
    """Turn an EvoPINNProblem instance into the uniform problem dictionary."""
    pretty = inst.name                     # e.g. 'EvoPINN1_Poisson1D'
    physics = pretty.split('_', 1)[1] if '_' in pretty else pretty
    return {
        'name': pretty,
        'description': (f'Physics-informed neural network for {physics}: '
                        f'minimise [PDE residual, boundary, initial/physics, data] losses'),
        'dim': int(inst.dim),
        'n_obj': int(inst.n_obj),
        'bounds': list(inst.bounds),
        'objective_types': list(inst.objective_types),
        'evaluate': (lambda x, _i=inst: np.asarray(_i.evaluate(np.asarray(x, dtype=float)),
                                                   dtype=float)),
        'true_front': None,                # no analytic PF -> approximated
        'objective_names': ['Residual_Loss', 'Boundary_Loss',
                            'Initial_Physics_Loss', 'Data_Loss'],
    }


_EVOPINN_CACHE = None  # module-level cache so EvoPINN problems are built once


def _evopinn_problems():
    """Lazily build (once) and cache the dict of all 12 EvoPINN problem
    dictionaries, keyed by their pretty name (e.g. 'EvoPINN1_Poisson1D')."""
    global _EVOPINN_CACHE
    if _EVOPINN_CACHE is None:
        insts = get_all_evopinn_problems()
        _EVOPINN_CACHE = {v.name: _wrap_evopinn(k, v)
                          for k, v in insts.items()}
    return _EVOPINN_CACHE


# Registry
EVOPINN_NAMES = ['EvoPINN1_Poisson1D', 'EvoPINN2_Heat1D', 'EvoPINN3_Advection1D',
                 'EvoPINN4_Wave1D', 'EvoPINN5_Burgers1D', 'EvoPINN6_AllenCahn1D',
                 'EvoPINN7_Poisson2D', 'EvoPINN8_ReactionDiffusion1D',
                 'EvoPINN9_DampedOscillator', 'EvoPINN10_VanDerPol',
                 'EvoPINN11_LotkaVolterra', 'EvoPINN12_Schrodinger1D']

ZDT_NAMES = ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6']
DTLZ_NAMES = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7']
MIXED_NAMES = ['Mixed_ZDT1', 'Mixed_DTLZ1', 'Mixed_DTLZ2']

PROBLEM_NAMES = EVOPINN_NAMES + ZDT_NAMES + DTLZ_NAMES + MIXED_NAMES


def get_problem(name):
    """Build (or fetch) one problem dictionary by name."""
    if name in STANDARD_BUILDERS:
        p = STANDARD_BUILDERS[name]()
    elif name in MIXED_BUILDERS:
        p = MIXED_BUILDERS[name]()
    else:
        evo = _evopinn_problems()
        if name not in evo:
            raise KeyError(f"unknown problem '{name}'. "
                           f"Valid names: {', '.join(PROBLEM_NAMES)}")
        p = evo[name]
    p.setdefault('objective_names', [f'Objective_{i+1}' for i in range(p['n_obj'])])
    return p


def get_all_problems(names=None):
    """Build every problem dictionary listed in `names` (defaults to all 27
    registered problems, in PROBLEM_NAMES order)."""
    names = list(names) if names else PROBLEM_NAMES
    return [get_problem(n) for n in names]


def bounds_arrays(problem):
    """Convenience: (lower, upper) as float arrays."""
    b = np.asarray(problem['bounds'], dtype=float)
    return b[:, 0].copy(), b[:, 1].copy()


def problem_table():
    """Small overview table (used by check.py and the README)."""
    rows = []
    for n in PROBLEM_NAMES:
        p = get_problem(n)
        rows.append({'Problem': p['name'], 'Dim': p['dim'],
                     'Objectives': p['n_obj'],
                     'Directions': ','.join(p['objective_types']),
                     'Analytic_PF': p['true_front'] is not None})
    return rows
