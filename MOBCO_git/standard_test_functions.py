"""
standard_test_functions.py -- Analytic objective functions for the ZDT
(2-objective) and DTLZ (M=3-objective) benchmark suites, all standard
minimization problems. Each builder function (zdt1, dtlz2, etc.) returns a
problem dict (see `_problem` below) that plugs directly into the rest of
the pipeline via dataset.py's STANDARD_BUILDERS registry.

Formulas and reference fronts for the ZDT family were checked against the
original Zitzler-Deb-Thiele (2000) paper.
"""

import numpy as np

import generate


def _problem(name, description, dim, n_obj, bounds, types, fn, front=None):
    """Package one benchmark problem into the common dict shape used
    throughout the codebase (dataset.py, optimizer.py, metrics.py, etc.)."""
    return {'name': name, 'description': description, 'dim': dim,
            'n_obj': n_obj, 'bounds': bounds, 'objective_types': types,
            'evaluate': fn, 'true_front': front}


# ZDT family  (2 objectives, both minimised)
def zdt1(dim=30):
    def f(x):
        x = np.asarray(x, dtype=float)
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    return _problem('ZDT1', 'Convex Pareto front, 2 objectives (both minimised)',
                    dim, 2, [(0.0, 1.0)] * dim, ['min', 'min'], f,
                    generate.true_front_zdt1)


def zdt2(dim=30):
    def f(x):
        x = np.asarray(x, dtype=float)
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2])
    return _problem('ZDT2', 'Non-convex Pareto front, 2 objectives (both minimised)',
                    dim, 2, [(0.0, 1.0)] * dim, ['min', 'min'], f,
                    generate.true_front_zdt2)


def zdt3(dim=30):
    def f(x):
        x = np.asarray(x, dtype=float)
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
        return np.array([f1, g * h])
    return _problem('ZDT3', 'Disconnected Pareto front, 2 objectives (both minimised)',
                    dim, 2, [(0.0, 1.0)] * dim, ['min', 'min'], f,
                    generate.true_front_zdt3)


def zdt4(dim=10):
    def f(x):
        x = np.asarray(x, dtype=float)
        f1 = x[0]
        tail = x[1:]
        g = 1.0 + 10.0 * (len(x) - 1) + np.sum(tail ** 2 - 10.0 * np.cos(4.0 * np.pi * tail))
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    bounds = [(0.0, 1.0)] + [(-5.0, 5.0)] * (dim - 1)
    return _problem('ZDT4', 'Multi-modal (21^9 local fronts), 2 objectives (both minimised)',
                    dim, 2, bounds, ['min', 'min'], f, generate.true_front_zdt4)


def zdt6(dim=10):
    def f(x):
        x = np.asarray(x, dtype=float)
        f1 = 1.0 - np.exp(-4.0 * x[0]) * np.sin(6.0 * np.pi * x[0]) ** 6
        g = 1.0 + 9.0 * (np.sum(x[1:]) / (len(x) - 1)) ** 0.25
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2])
    return _problem('ZDT6', 'Non-uniform, biased density, 2 objectives (both minimised)',
                    dim, 2, [(0.0, 1.0)] * dim, ['min', 'min'], f,
                    generate.true_front_zdt6)


# DTLZ family  (M = 3 objectives, all minimised)
def _g_dtlz1(xm):
    """DTLZ1's multi-modal 'g' distance function -- 0 at xm=0.5, with many
    regularly-spaced local optima (the cosine term) around the true PS."""
    return 100.0 * (len(xm) + np.sum((xm - 0.5) ** 2 -
                                     np.cos(20.0 * np.pi * (xm - 0.5))))


def _g_sphere(xm):
    """Simple unimodal 'g' distance function shared by DTLZ2/4/5 -- 0 at
    xm=0.5, a smooth sphere elsewhere (no local optima)."""
    return np.sum((xm - 0.5) ** 2)


def dtlz1(m=3, k=5):
    dim = m - 1 + k

    def f(x):
        x = np.asarray(x, dtype=float)
        xm = x[m - 1:]
        g = _g_dtlz1(xm)
        out = np.empty(m)
        for i in range(m):
            v = 0.5 * (1.0 + g)
            for j in range(m - 1 - i):
                v *= x[j]
            if i > 0:
                v *= (1.0 - x[m - 1 - i])
            out[i] = v
        return out
    return _problem('DTLZ1', 'Linear simplex front, 3 objectives (all minimised)',
                    dim, m, [(0.0, 1.0)] * dim, ['min'] * m, f,
                    generate.true_front_dtlz1)


def _dtlz_sphere_eval(x, m, g):
    """Shared M-objective spherical-front evaluation used by DTLZ2/3/4:
    f_i = (1+g) * prod(cos of the first m-1-i position variables) *
    (sin of the next one, if i > 0). x[:m-1] are the "position" variables
    (which shape where on the front the solution lands); g comes from the
    remaining "distance" variables x[m-1:]."""
    out = np.empty(m)
    for i in range(m):
        v = 1.0 + g
        for j in range(m - 1 - i):
            v *= np.cos(0.5 * np.pi * x[j])
        if i > 0:
            v *= np.sin(0.5 * np.pi * x[m - 1 - i])
        out[i] = v
    return out


def dtlz2(m=3, k=10):
    dim = m - 1 + k

    def f(x):
        x = np.asarray(x, dtype=float)
        return _dtlz_sphere_eval(x, m, _g_sphere(x[m - 1:]))
    return _problem('DTLZ2', 'Spherical front, 3 objectives (all minimised)',
                    dim, m, [(0.0, 1.0)] * dim, ['min'] * m, f,
                    generate.true_front_dtlz2)


def dtlz3(m=3, k=10):
    dim = m - 1 + k

    def f(x):
        x = np.asarray(x, dtype=float)
        return _dtlz_sphere_eval(x, m, _g_dtlz1(x[m - 1:]))
    return _problem('DTLZ3', 'Spherical front + many local fronts, 3 objectives (all minimised)',
                    dim, m, [(0.0, 1.0)] * dim, ['min'] * m, f,
                    generate.true_front_dtlz3)


def dtlz4(m=3, k=10, alpha=100.0):
    dim = m - 1 + k

    def f(x):
        x = np.asarray(x, dtype=float).copy()
        x[:m - 1] = x[:m - 1] ** alpha
        return _dtlz_sphere_eval(x, m, _g_sphere(x[m - 1:]))
    return _problem('DTLZ4', 'Spherical front with biased density, 3 objectives (all minimised)',
                    dim, m, [(0.0, 1.0)] * dim, ['min'] * m, f,
                    generate.true_front_dtlz4)


def _dtlz56_eval(x, m, g):
    """Shared evaluation for DTLZ5/6's degenerate-curve front: unlike
    DTLZ2-4, the angular parameters `theta` here are themselves a function
    of g and x, which collapses the front from an (m-1)-dimensional surface
    down to a 1D curve."""
    theta = np.empty(m - 1)
    theta[0] = 0.5 * np.pi * x[0]
    if m > 2:
        t = np.pi / (4.0 * (1.0 + g))
        theta[1:] = t * (1.0 + 2.0 * g * x[1:m - 1])
    out = np.empty(m)
    for i in range(m):
        v = 1.0 + g
        for j in range(m - 1 - i):
            v *= np.cos(theta[j])
        if i > 0:
            v *= np.sin(theta[m - 1 - i])
        out[i] = v
    return out


def dtlz5(m=3, k=10):
    dim = m - 1 + k

    def f(x):
        x = np.asarray(x, dtype=float)
        return _dtlz56_eval(x, m, _g_sphere(x[m - 1:]))
    return _problem('DTLZ5', 'Degenerate curve front, 3 objectives (all minimised)',
                    dim, m, [(0.0, 1.0)] * dim, ['min'] * m, f,
                    generate.true_front_dtlz5)


def dtlz6(m=3, k=10):
    dim = m - 1 + k

    def f(x):
        x = np.asarray(x, dtype=float)
        xm = x[m - 1:]
        g = np.sum(xm ** 0.1)
        return _dtlz56_eval(x, m, g)
    return _problem('DTLZ6', 'Degenerate curve front, hard convergence, 3 objectives (all minimised)',
                    dim, m, [(0.0, 1.0)] * dim, ['min'] * m, f,
                    generate.true_front_dtlz6)


def dtlz7(m=3, k=20):
    dim = m - 1 + k

    def f(x):
        x = np.asarray(x, dtype=float)
        xm = x[m - 1:]
        g = 1.0 + 9.0 * np.mean(xm)
        out = np.empty(m)
        out[:m - 1] = x[:m - 1]
        h = m - np.sum(out[:m - 1] / (1.0 + g) *
                       (1.0 + np.sin(3.0 * np.pi * out[:m - 1])))
        out[m - 1] = (1.0 + g) * h
        return out
    return _problem('DTLZ7', 'Four disconnected front patches, 3 objectives (all minimised)',
                    dim, m, [(0.0, 1.0)] * dim, ['min'] * m, f,
                    generate.true_front_dtlz7)


STANDARD_BUILDERS = {   # Registry consumed by dataset.py to build each named problem on demand
    'ZDT1': zdt1, 'ZDT2': zdt2, 'ZDT3': zdt3, 'ZDT4': zdt4, 'ZDT6': zdt6,
    'DTLZ1': dtlz1, 'DTLZ2': dtlz2, 'DTLZ3': dtlz3, 'DTLZ4': dtlz4,
    'DTLZ5': dtlz5, 'DTLZ6': dtlz6, 'DTLZ7': dtlz7,
}
