"""
check.py -- Lightweight, fast self-check / smoke-test suite for the whole
project. Not a formal unit-test framework (no pytest/unittest) -- each
check_* function prints [ok]/[FAIL] lines and returns a count of failures;
`main()` sums them up and reports overall pass/fail. Meant to be run quickly
before committing to a full (expensive) 27-problem x 50-run experiment, to
catch obvious regressions first.
"""

import numpy as np

from config import MOBCOConfig
from dataset import PROBLEM_NAMES, get_problem, bounds_arrays, problem_table
from dominance import dominates, nondominated_mask
from metrics import MetricContext, evaluate_front
from optimizer import run_single

OK, FAIL = '  [ok]  ', '  [FAIL]'


def check_problems():
    """Every registered problem: evaluate one random feasible point and
    confirm dimensions/objective count/finiteness all line up with what the
    problem dict declares."""
    print('1-2. building and evaluating all problems')
    rng = np.random.default_rng(0)
    bad = 0
    for row in problem_table():
        p = get_problem(row['Problem'])
        lo, hi = bounds_arrays(p)
        x = lo + rng.random(p['dim']) * (hi - lo)
        f = p['evaluate'](x)
        ok = (len(lo) == p['dim'] and len(f) == p['n_obj']
              and np.all(np.isfinite(f))
              and len(p['objective_types']) == p['n_obj'])
        bad += (not ok)
        print(f"{OK if ok else FAIL} {p['name']:<28} dim={p['dim']:<4} "
              f"M={p['n_obj']}  dirs={','.join(p['objective_types']):<15} "
              f"f={np.array2string(f, precision=3, max_line_width=200)}")
    return bad


def check_dominance():
    """Verify the direction-aware dominance logic on a few hand-picked
    vectors under a mixed (min, max) objective setup."""
    print('\n3. dominance with a mixed direction set  (min, max)')
    types = ['min', 'max']
    a = np.array([1.0, 5.0])      # lower f1 AND higher f2 -> better on both
    b = np.array([2.0, 3.0])
    t1 = dominates(a, b, types) and not dominates(b, a, types)
    print(f"{OK if t1 else FAIL} [1,5] dominates [2,3] under (min,max)")

    c = np.array([1.0, 2.0])      # better f1, worse f2 -> mutually non-dominated
    d = np.array([2.0, 4.0])
    t2 = (not dominates(c, d, types)) and (not dominates(d, c, types))
    print(f"{OK if t2 else FAIL} [1,2] and [2,4] are mutually non-dominated")

    F = np.array([[1.0, 5.0], [2.0, 3.0], [0.5, 1.0], [3.0, 9.0]])
    mask = nondominated_mask(F, types)
    t3 = bool(mask[0] and mask[2] and mask[3] and not mask[1])
    print(f"{OK if t3 else FAIL} non-dominated mask = {mask.astype(int)}")
    return int(not (t1 and t2 and t3))


def check_no_conversion():
    """Confirm that 'max' objectives in the Mixed_* problems are genuinely
    maximization quantities stored as-is (no implicit sign-flip to force
    everything into a minimization form) -- this matches the project's
    direction-aware dominance design (see dominance.py)."""
    print('\n4. maximisation objectives are stored unconverted')
    bad = 0
    for name in ('Mixed_ZDT1', 'Mixed_DTLZ1', 'Mixed_DTLZ2'):
        p = get_problem(name)
        lo, hi = bounds_arrays(p)
        rng = np.random.default_rng(7)
        F = np.array([p['evaluate'](lo + rng.random(p['dim']) * (hi - lo))
                      for _ in range(200)])
        for j, t in enumerate(p['objective_types']):
            if t != 'max':
                continue
            positive = bool(np.all(F[:, j] >= -1e-12))
            bad += (not positive)
            print(f"{OK if positive else FAIL} {name} f{j+1} is a genuine "
                  f"'max' quantity, range "
                  f"[{F[:, j].min():.4f}, {F[:, j].max():.4f}] (no sign flip)")
    return bad


def check_reference_fronts():
    """For every problem with an analytic true_front generator, sample it and
    confirm the sampled points are (almost entirely) mutually non-dominated
    -- i.e. the reference front generator itself is producing a valid Pareto
    front, not something with obvious dominated points in it."""
    print('\n5. analytic reference fronts are internally non-dominated')
    bad = 0
    for name in PROBLEM_NAMES:
        p = get_problem(name)
        if p['true_front'] is None:
            continue
        R = p['true_front'](300)
        frac = nondominated_mask(R, p['objective_types']).mean()
        ok = frac > 0.98
        bad += (not ok)
        print(f"{OK if ok else FAIL} {p['name']:<14} "
              f"{frac * 100:5.1f}% of {len(R)} reference points non-dominated")
    return bad


def check_optimizer():
    """Run MOBCO briefly on four representative problems (2-obj, mixed-
    direction, 3-obj) and confirm the archive's hypervolume improves from the
    first recorded snapshot to the final front -- a basic sanity check that
    the optimizer is actually optimizing, not just producing random points."""
    print('\n6. short MOBCO run improves the hypervolume')
    cfg = MOBCOConfig(n_population=40, n_dogs=3, archive_size=40,
                      max_iterations=25, n_runs=1, history_metric_runs=1,
                      hv_mc_samples_history=2000)
    bad = 0
    for name in ('ZDT1', 'Mixed_ZDT1', 'DTLZ2', 'Mixed_DTLZ2'):
        p = get_problem(name)
        out = run_single(p, cfg, seed=1)
        ctx = MetricContext(p['true_front'](400), p['objective_types'])
        first = evaluate_front(out['history']['snapshots'][0], ctx, 4000)['hv']
        last = evaluate_front(out['front'], ctx, 8000)['hv']
        ok = last >= first
        bad += (not ok)
        print(f"{OK if ok else FAIL} {name:<12} HV {first:.4f} -> {last:.4f} "
              f"({len(out['front'])} front points, {out['runtime']:.2f}s)")
    return bad


def main():
    """Run every check_* function in sequence and report a combined pass/fail
    summary. Returns 0 (success) or 1 (at least one check failed), suitable
    as a process exit code."""
    print('MOBCO project self-check\n' + '=' * 70)
    bad = 0
    bad += check_problems()
    bad += check_dominance()
    bad += check_no_conversion()
    bad += check_reference_fronts()
    bad += check_optimizer()
    print('\n' + '=' * 70)
    print('ALL CHECKS PASSED' if bad == 0 else f'{bad} CHECK(S) FAILED')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
