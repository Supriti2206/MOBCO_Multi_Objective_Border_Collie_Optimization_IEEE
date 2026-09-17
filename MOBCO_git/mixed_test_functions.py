"""
mixed_test_functions.py — Mixed_ZDT1, Mixed_DTLZ1, Mixed_DTLZ2.

Mixed_ZDT1   (dim 30, x in [0,1]^30)
    g  = 1 + 9 * mean(x2..xn)                      (distance term, ideally 1)
    f1 = x1                              -> MIN    "cost"
    f2 = sqrt(x1) / g                    -> MAX    "yield"
    Conflict: f1 wants x1 -> 0, f2 wants x1 -> 1.  Both improve as g -> 1.
    Pareto front (g = 1):   f2 = sqrt(f1),  f1 in [0, 1]

Mixed_DTLZ1  (M = 3, dim 7, x in [0,1]^7)
    g  = DTLZ1 multi-modal distance term (ideally 0)
    f1 = 0.5 * x1 * x2       * (1 + g)   -> MIN
    f2 = 0.5 * x1 * (1 - x2) * (1 + g)   -> MIN
    f3 = 0.5 * x1 / (1 + g)              -> MAX    "throughput"
    Conflict: f1, f2 want x1 -> 0, f3 wants x1 -> 1.
    Pareto front (g = 0): the triangle  f3 = f1 + f2,  f1,f2 >= 0, f1+f2 <= 0.5

Mixed_DTLZ2  (M = 3, dim 12, x in [0,1]^12)
    g  = sum (xi - 0.5)^2                          (ideally 0)
    f1 = (1 + g) * cos(x1*pi/2) * cos(x2*pi/2)   -> MIN
    f2 = (1 + g) * cos(x1*pi/2) * sin(x2*pi/2)   -> MIN
    f3 = cos(x1*pi/2) / (1 + g)                  -> MAX   "signal strength"
    Conflict: f1, f2 want cos(x1*pi/2) -> 0, f3 wants it -> 1.
    Pareto front (g = 0): the cone  f3 = sqrt(f1^2 + f2^2),  f3 in [0, 1]
"""

import numpy as np

import generate
from standard_test_functions import _problem, _g_dtlz1


def mixed_zdt1(dim=30):
    def f(x):
        x = np.asarray(x, dtype=float)
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        f1 = x[0]                       # minimise
        f2 = np.sqrt(x[0]) / g          # maximise
        return np.array([f1, f2])
    p = _problem('Mixed_ZDT1',
                    'Mixed directions: f1 minimised (cost), f2 maximised (yield)',
                    dim, 2, [(0.0, 1.0)] * dim, ['min', 'max'], f,
                    generate.true_front_mixed_zdt1)
    p['objective_names'] = ['Cost', 'Yield']
    return p


def mixed_dtlz1(m=3, k=5):
    dim = m - 1 + k

    def f(x):
        x = np.asarray(x, dtype=float)
        g = _g_dtlz1(x[m - 1:])
        f1 = 0.5 * x[0] * x[1] * (1.0 + g)          # minimise
        f2 = 0.5 * x[0] * (1.0 - x[1]) * (1.0 + g)  # minimise
        f3 = 0.5 * x[0] / (1.0 + g)                 # maximise
        return np.array([f1, f2, f3])
    p = _problem('Mixed_DTLZ1',
                    'Mixed directions: f1, f2 minimised, f3 maximised (throughput); multi-modal g',
                    dim, m, [(0.0, 1.0)] * dim, ['min', 'min', 'max'], f,
                    generate.true_front_mixed_dtlz1)
    p['objective_names'] = ['Cost_A', 'Cost_B', 'Throughput']
    return p


def mixed_dtlz2(m=3, k=10):
    dim = m - 1 + k

    def f(x):
        x = np.asarray(x, dtype=float)
        g = np.sum((x[m - 1:] - 0.5) ** 2)
        c1 = np.cos(0.5 * np.pi * x[0])
        f1 = (1.0 + g) * c1 * np.cos(0.5 * np.pi * x[1])   # minimise
        f2 = (1.0 + g) * c1 * np.sin(0.5 * np.pi * x[1])   # minimise
        f3 = c1 / (1.0 + g)                                # maximise
        return np.array([f1, f2, f3])
    p = _problem('Mixed_DTLZ2',
                    'Mixed directions: f1, f2 minimised, f3 maximised (signal strength)',
                    dim, m, [(0.0, 1.0)] * dim, ['min', 'min', 'max'], f,
                    generate.true_front_mixed_dtlz2)
    p['objective_names'] = ['Interference_A', 'Interference_B', 'Signal_Strength']
    return p


MIXED_BUILDERS = {
    'Mixed_ZDT1': mixed_zdt1,
    'Mixed_DTLZ1': mixed_dtlz1,
    'Mixed_DTLZ2': mixed_dtlz2,
}
