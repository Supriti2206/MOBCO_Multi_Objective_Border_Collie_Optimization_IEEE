"""
Requires: pip install SALib

Split into TWO independent 5-parameter Sobol studies instead of one
10-parameter study. Cost scales with (num_vars + 2), so two 5-var
studies are much cheaper per sample than one 10-var study, and
rankings within each smaller group converge faster for the same
compute budget. While one group is being varied, the other group is
frozen at MOBCOConfig's nominal/default values.

  Study "structural" (matches the original paper's scope most closely):
      n_population, n_dogs, mutation_rate, archive_size, eyeing_steps

  Study "shaping" (this codebase's analogue of the paper's
  "velocity, time" parameters -- Vt/acc/t are recomputed every
  iteration from formulas, so instead of varying them directly we
  vary the constants in velocity_time_acceleration.py that govern
  their scale and decay):
      acc_damping, mutation_decay_floor, step_decay_floor,
      stalking_pull (c1), gathering_pull (c2)

What it does
------------
1. Defines each study's parameters as a Sobol problem (bounds), with
   the other study's parameters frozen at MOBCOConfig defaults.
2. Uses Saltelli/Sobol sampling to build a parameter matrix.
3. For every sampled parameter set, runs MOBCO (a few short, cheap
   replicate seeds) on a small set of representative benchmark problems
   and records the mean final Hypervolume as the model output Y.
4. Runs Sobol.analyze() to get first-order (S1) and total-order (ST)
   indices for every parameter -- higher index = bigger influence on HV.
5. Prints a table per study (mirrors Table 8 in the paper) and saves
   both to results/sensitivity_analysis_<study>.xlsx

Usage
-----
python sensitivity_analysis.py --study structural --base-n 16
python sensitivity_analysis.py --study shaping --base-n 16
python sensitivity_analysis.py --study both --base-n 16     # runs both, back to back
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    # newer SALib (>=1.5) renamed this module
    from SALib.sample.sobol import sample as sobol_sample
except ImportError:
    from SALib.sample.saltelli import sample as sobol_sample
from SALib.analyze import sobol

from config import MOBCOConfig
from dataset import get_problem
from metrics import MetricContext, hypervolume
from optimizer import MOBCO

ROOT = Path(__file__).resolve().parent

# Nominal/default values every parameter is frozen at when it's NOT the
# study currently being varied. Taken straight from MOBCOConfig() defaults.
_DEFAULTS = MOBCOConfig()

STUDIES = {
    'structural': {
        'num_vars': 5,
        'names': ['n_population', 'n_dogs', 'mutation_rate',
                  'archive_size', 'eyeing_steps'],
        'bounds': [
            [20, 350],     # n_population  (widened so final=300 sits inside)
            [2, 8],        # n_dogs
            [0.01, 0.4],   # mutation_rate
            [20, 350],     # archive_size  (widened so final=300 sits inside)
            [2, 12],       # eyeing_steps
        ],
        'is_int': [True, True, False, True, True],
    },
    'shaping': {
        'num_vars': 5,
        'names': ['acc_damping', 'mutation_decay_floor', 'step_decay_floor',
                  'stalking_pull', 'gathering_pull'],
        'bounds': [
            [0.5, 0.99],   # acc_damping
            [0.0, 0.5],    # mutation_decay_floor
            [0.0, 0.5],    # step_decay_floor
            [0.5, 4.0],    # stalking_pull  (c1)
            [0.1, 2.0],    # gathering_pull (c2)
        ],
        'is_int': [False, False, False, False, False],
    },
}


def _cast_row(row, study):
    """Round integer-valued parameters sampled from a continuous range,
    then merge with the frozen defaults for the OTHER study's params."""
    params = {
        'n_population': _DEFAULTS.n_population,
        'n_dogs': _DEFAULTS.n_dogs,
        'mutation_rate': _DEFAULTS.mutation_rate,
        'archive_size': _DEFAULTS.archive_size,
        'eyeing_steps': _DEFAULTS.eyeing_steps,
        'acc_damping': _DEFAULTS.acc_damping,
        'mutation_decay_floor': _DEFAULTS.mutation_decay_floor,
        'step_decay_floor': _DEFAULTS.step_decay_floor,
        'stalking_pull': _DEFAULTS.stalking_pull,
        'gathering_pull': _DEFAULTS.gathering_pull,
    }
    spec = STUDIES[study]
    for name, is_int, value in zip(spec['names'], spec['is_int'], row):
        if name == 'n_dogs':
            params[name] = max(2, int(round(value)))
        elif name == 'eyeing_steps':
            params[name] = max(1, int(round(value)))
        elif is_int:
            params[name] = int(round(value))
        else:
            params[name] = float(value)
    return params


def _make_config(params, sa_iterations, sa_seed):
    """Build a MOBCOConfig for one sensitivity-analysis trial: start from
    the nominal defaults, override with this sampled parameter set, and
    force a small max_iterations / fixed base_seed / quiet mode (SA needs
    many cheap runs, not full-length experiments)."""
    cfg = MOBCOConfig()
    for k, v in params.items():
        setattr(cfg, k, v)
    cfg.max_iterations = sa_iterations
    cfg.verbose = False
    cfg.base_seed = sa_seed
    return cfg


def _final_hv(problem, cfg, seed):
    """Run one MOBCO trial and return its final (normalised) hypervolume."""
    res = MOBCO(problem, cfg, seed=seed).run()
    front = res['front']
    if front.size == 0:
        return 0.0
    tf = problem.get('true_front')
    if tf is not None:
        # true_front is a GENERATOR FUNCTION (e.g. generate.true_front_zdt1),
        # not a precomputed array -- call it to get n reference points,
        # same pattern run_experiment.py uses.
        ref_front = np.asarray(tf(1000), dtype=float)
    else:
        # no analytic PF (e.g. EvoPINN problems): use the archive itself
        # as its own reference so HV is still comparable across params
        ref_front = front
    if ref_front is None or len(ref_front) == 0:
        ref_front = front
    ctx = MetricContext(ref_front, problem['objective_types'], seed=seed)
    hv = hypervolume(front, ctx, n_samples=2000)
    return 0.0 if np.isnan(hv) else hv


def evaluate_row(row, study, problems, sa_iterations, sa_replicates):
    """Model output Y for one sampled parameter set: mean HV across
    problems and replicate seeds (averaging tames MOBCO's stochasticity)."""
    params = _cast_row(row, study)
    hv_values = []
    for pname in problems:
        problem = get_problem(pname)
        cfg = _make_config(params, sa_iterations, sa_seed=2000)
        for r in range(sa_replicates):
            hv_values.append(_final_hv(problem, cfg, seed=cfg.base_seed + r))
    return float(np.mean(hv_values))


def run_study(study, base_n, sa_iterations, sa_replicates, problems):
    """
    Run one full Sobol sensitivity study end to end: build the Sobol sample
    matrix, evaluate every sampled parameter combination (mean HV across
    problems/replicates), compute first- and total-order Sobol indices, and
    save both the ranked-indices table and the raw per-run data to Excel.
    """
    problem_def = {'num_vars': STUDIES[study]['num_vars'],
                   'names': STUDIES[study]['names'],
                   'bounds': STUDIES[study]['bounds']}

    param_values = sobol_sample(problem_def, base_n, calc_second_order=False)
    n_runs = len(param_values)
    print(f"\n[{study}] Sobol SA: {n_runs} MOBCO calls "
          f"({sa_iterations} iters x {sa_replicates} seeds x "
          f"{len(problems)} problems each; other 5 params frozen at defaults)")

    Y = np.empty(n_runs)
    t0 = time.perf_counter()
    for i, row in enumerate(param_values, 1):
        Y[i - 1] = evaluate_row(row, study, problems, sa_iterations, sa_replicates)
        if i % max(1, n_runs // 20) == 0 or i == n_runs:
            elapsed = time.perf_counter() - t0
            print(f"  [{study}] {i}/{n_runs}  (elapsed {elapsed/60:.1f} min)")

    Si = sobol.analyze(problem_def, Y, calc_second_order=False, print_to_console=False)

    # Save the RAW per-run data too: every sampled parameter combo + the
    # HV it achieved. This is what you actually need for Step 5 (finding
    # the best VALUE for an influential parameter, not just how much it
    # matters) -- sort this file by hv_score to see which parameter
    # values cluster among the best-performing runs.
    raw_table = pd.DataFrame(param_values, columns=problem_def['names'])
    raw_table['hv_score'] = Y
    raw_table = raw_table.sort_values('hv_score', ascending=False)
    raw_out_path = ROOT / 'results' / f'sensitivity_raw_runs_{study}.xlsx'
    (ROOT / 'results').mkdir(exist_ok=True)
    raw_table.to_excel(raw_out_path, index=False)
    print(f"saved raw per-run data -> {raw_out_path}")

    table = pd.DataFrame({
        'parameter': problem_def['names'],
        'S1_first_order': Si['S1'],
        'ST_total_order': Si['ST'],
        'S1_conf': Si['S1_conf'],
        'ST_conf': Si['ST_conf'],
    }).sort_values('ST_total_order', ascending=False)

    print(f"\n=== [{study}] Sobol sensitivity indices (higher = more influence on HV) ===")
    print(table.to_string(index=False))

    out_dir = ROOT / 'results'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'sensitivity_analysis_{study}.xlsx'
    table.to_excel(out_path, index=False)
    print(f"saved -> {out_path}")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', choices=['structural', 'shaping', 'both'],
                    default='both',
                    help='which 5-parameter group to analyze')
    ap.add_argument('--base-n', type=int, default=16,
                    help='Saltelli/Sobol base sample size N '
                         '(total runs per study = N * (num_vars + 2); '
                         'num_vars=5 per study, so N=16 -> 112 runs/study)')
    ap.add_argument('--sa-iterations', type=int, default=50,
                    help='MOBCO max_iterations during SA (keep small)')
    ap.add_argument('--sa-replicates', type=int, default=3,
                    help='seeds averaged per parameter set')
    ap.add_argument('--problems', nargs='*',
                    default=['ZDT1', 'DTLZ2', 'EvoPINN1_Poisson1D'])
    args = ap.parse_args()

    studies = ['structural', 'shaping'] if args.study == 'both' else [args.study]
    for study in studies:
        run_study(study, args.base_n, args.sa_iterations,
                  args.sa_replicates, args.problems)


if __name__ == '__main__':
    main()