"""
run_experiment.py -- Orchestrates a full multi-run experiment for a single
problem: runs MOBCO n_runs times, builds a shared reference front, scores
every run's front with metrics.py, and aggregates statistics across runs.
This is what main.py calls once per problem.
"""

import numpy as np

from generate import approximate_reference_front
from metrics import MetricContext, evaluate_front, hypervolume, \
    inverted_generational_distance, generational_distance
from nondominated_sort import pareto_front, unique_rows
from optimizer import run_single


def build_reference_front(problem, config, run_outputs):
    """Analytic PF if the problem has one, else the union of all archives.

    Problems with a known closed-form Pareto front (ZDT/DTLZ/Mixed_*) use
    that directly for a fair, ground-truth comparison. Problems without one
    (the EvoPINN problems) instead approximate a reference front from the
    combined (non-dominated) archive across every run of this experiment --
    the best available stand-in when the true front is unknown."""
    tf = problem.get('true_front')
    if tf is not None:
        R = np.asarray(tf(config.ref_front_size), dtype=float)
        R, _ = unique_rows(R)
        return R, 'analytic'
    R = approximate_reference_front([o['front'] for o in run_outputs],
                                    problem['objective_types'],
                                    max_size=config.ref_front_size)
    return R, 'approximated from all runs'


def _history_metrics(run_outputs, ctx, config):
    """Per-generation HV / IGD / GD / sizes, averaged over the traced runs.

    Only a subset of runs (config.history_metric_runs) keep their full
    per-generation snapshots (see run_problem below) -- this computes the
    convergence trace (mean +/- std across those traced runs) used for the
    convergence plots, without re-running metrics on every single run's full
    history (which would be expensive across n_runs x max_iterations)."""
    traced = [o for o in run_outputs if o['history'].get('snapshots')]
    if not traced:
        return None
    iters = traced[0]['history']['iteration']
    n_pts = len(iters)
    hv = np.full((len(traced), n_pts), np.nan)
    igd = np.full((len(traced), n_pts), np.nan)
    gd = np.full((len(traced), n_pts), np.nan)
    arch = np.full((len(traced), n_pts), np.nan)
    front = np.full((len(traced), n_pts), np.nan)
    best = np.full((len(traced), n_pts, ctx.R.shape[1]), np.nan)
    mean = np.full((len(traced), n_pts, ctx.R.shape[1]), np.nan)

    for r, out in enumerate(traced):
        h = out['history']
        for t, F in enumerate(h['snapshots']):
            if len(F) == 0:
                continue
            hv[r, t] = hypervolume(F, ctx, config.hv_mc_samples_history)
            igd[r, t] = inverted_generational_distance(F, ctx)
            gd[r, t] = generational_distance(F, ctx)
        arch[r] = h['archive_size']
        front[r] = h['front_size']
        best[r] = np.asarray(h['best'])
        mean[r] = np.asarray(h['mean'])

    with np.errstate(invalid='ignore'):  # nanmean/nanstd warn on all-NaN slices; expected/benign here
        return {
            'iteration': np.asarray(iters),
            'hv': np.nanmean(hv, axis=0), 'igd': np.nanmean(igd, axis=0),
            'gd': np.nanmean(gd, axis=0),
            'hv_std': np.nanstd(hv, axis=0), 'igd_std': np.nanstd(igd, axis=0),
            'gd_std': np.nanstd(gd, axis=0),
            'archive_size': np.nanmean(arch, axis=0),
            'front_size': np.nanmean(front, axis=0),
            'best': np.nanmean(best, axis=0), 'mean': np.nanmean(mean, axis=0),
            'n_traced_runs': len(traced),
        }


def _aggregate(run_metrics, runtimes):
    """Collapse the per-run metric dicts (from evaluate_front) into
    mean/std/median/best/worst summary statistics across all runs, plus
    runtime statistics -- this is what feeds the paper's summary tables."""
    keys = ['hv', 'igd', 'gd', 'spacing', 'spread', 'epsilon', 'size']
    agg = {}
    for k in keys:
        v = np.array([m[k] for m in run_metrics], dtype=float)
        v = v[np.isfinite(v)]  # drop any NaN runs (e.g. an empty front) before aggregating
        if v.size == 0:
            agg[f'{k}_mean'] = agg[f'{k}_std'] = np.nan
            agg[f'{k}_median'] = agg[f'{k}_best'] = agg[f'{k}_worst'] = np.nan
            continue
        agg[f'{k}_mean'] = float(np.mean(v))
        agg[f'{k}_std'] = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
        agg[f'{k}_median'] = float(np.median(v))
        larger_better = (k in ('hv', 'size'))  # HV and front size: bigger is better; everything else: smaller is better
        agg[f'{k}_best'] = float(np.max(v) if larger_better else np.min(v))
        agg[f'{k}_worst'] = float(np.min(v) if larger_better else np.max(v))
    rt = np.asarray(runtimes, dtype=float)
    agg['runtime_mean'] = float(rt.mean())
    agg['runtime_std'] = float(rt.std(ddof=1)) if rt.size > 1 else 0.0
    agg['runtime_total'] = float(rt.sum())
    return agg


def run_problem(problem, config, log=print):
    """
    Run the full n_runs-run experiment for one problem and return a results
    dict bundling everything downstream code (excel_writer.py,
    visualization.py) needs: per-run outputs/metrics, aggregated statistics,
    the convergence history, the shared reference front, and the combined
    (union-of-all-runs) Pareto front.
    """
    name = problem['name']
    log(f"\n=== {name} | dim={problem['dim']} | "
        f"objectives={problem['n_obj']} "
        f"({', '.join(problem['objective_types'])}) ===")

    run_outputs = []
    for r in range(config.n_runs):
        seed = config.base_seed + r  # deterministic per-run seed for reproducibility
        out = run_single(problem, config, seed)
        if r >= config.history_metric_runs:
            out['history']['snapshots'] = []      # free memory (only the first
                                                    # history_metric_runs runs
                                                    # keep their full trace)
        run_outputs.append(out)
        if config.verbose:
            log(f"  run {r + 1:>3}/{config.n_runs}  "
                f"front={len(out['front']):>4}  "
                f"archive={len(out['archive_fitness']):>4}  "
                f"{out['runtime']:.2f}s")

    R, ref_kind = build_reference_front(problem, config, run_outputs)
    log(f"  reference front: {len(R)} points ({ref_kind})")

    # One shared MetricContext for this problem -- reused across every run so
    # HV/IGD/etc. are computed on identical scaling (see metrics.py)
    ctx = MetricContext(R, problem['objective_types'], margin=config.hv_margin)

    run_metrics = [evaluate_front(o['front'], ctx, config.hv_mc_samples_final)
                   for o in run_outputs]
    for m, o in zip(run_metrics, run_outputs):
        m['runtime'] = o['runtime']

    history = _history_metrics(run_outputs, ctx, config)
    aggregate = _aggregate(run_metrics, [o['runtime'] for o in run_outputs])

    # best run = highest hypervolume
    hv_vals = [m['hv'] if np.isfinite(m['hv']) else -np.inf for m in run_metrics]
    best_run = int(np.argmax(hv_vals))

    # combined front over every run ("reality" in the comparison plots) --
    # union of all runs' final fronts, re-filtered down to non-dominated
    all_F = np.vstack([o['front'] for o in run_outputs if len(o['front'])])
    combined_front = pareto_front(all_F, objective_types=problem['objective_types'])

    log(f"  HV {aggregate['hv_mean']:.4f} +/- {aggregate['hv_std']:.4f} | "
        f"IGD {aggregate['igd_mean']:.4f} | GD {aggregate['gd_mean']:.4f} | "
        f"best run #{best_run + 1}")

    return {
        'problem': problem,
        'config': config,
        'run_outputs': run_outputs,
        'run_metrics': run_metrics,
        'aggregate': aggregate,
        'history': history,
        'reference_front': R,
        'reference_kind': ref_kind,
        'metric_context': ctx,
        'best_run': best_run,
        'combined_front': combined_front,
    }
