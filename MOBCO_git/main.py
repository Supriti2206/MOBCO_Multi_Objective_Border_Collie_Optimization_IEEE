"""
main.py -- Command-line entry point for the full MOBCO benchmark suite.

Typical usage:
    python main.py                       # run all 27 problems, full config
    python main.py --quick               # fast smoke test (QUICK config)
    python main.py --problems ZDT1 ZDT2  # only run a subset of problems

For each requested problem this: runs the experiment (run_problem), writes a
per-problem Excel workbook and figures, then writes a master summary workbook
and an overview figure across all problems.
"""

import argparse
import sys
import time
from pathlib import Path

from config import MOBCOConfig, QUICK
from dataset import PROBLEM_NAMES, get_problem
from excel_writer import write_problem_workbook, write_master_summary
from run_experiment import run_problem
from visualization import make_all_figures, plot_overview

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    """Define and parse the CLI flags (each maps to a MOBCOConfig field
    override -- see build_config below)."""
    ap = argparse.ArgumentParser(description='MOBCO multi-objective benchmark suite')
    ap.add_argument('--quick', action='store_true',
                    help='fast smoke test configuration')
    ap.add_argument('--problems', nargs='*', default=None,
                    help='subset of problem names (default: all 27 in dataset.py)')
    ap.add_argument('--runs', type=int, default=None)
    ap.add_argument('--iterations', type=int, default=None)
    ap.add_argument('--population', type=int, default=None)
    ap.add_argument('--dogs', type=int, default=None)
    ap.add_argument('--archive', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    return ap.parse_args(argv)


def build_config(args):
    """Start from QUICK (if --quick) or the full DEFAULT-equivalent
    MOBCOConfig(), then apply any explicit CLI overrides on top."""
    cfg = QUICK if args.quick else MOBCOConfig()
    if args.runs is not None:
        cfg.n_runs = args.runs
    if args.iterations is not None:
        cfg.max_iterations = args.iterations
    if args.population is not None:
        cfg.n_population = args.population
    if args.dogs is not None:
        cfg.n_dogs = args.dogs
    if args.archive is not None:
        cfg.archive_size = args.archive
    if args.seed is not None:
        cfg.base_seed = args.seed
    if args.quiet:
        cfg.verbose = False
    if args.problems:
        cfg.problems = list(args.problems)
    return cfg


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)
    results_dir, figures_dir = cfg.ensure_dirs(ROOT)

    names = cfg.problems if cfg.problems else PROBLEM_NAMES
    unknown = [n for n in names if n not in PROBLEM_NAMES]
    if unknown:
        print(f"Unknown problem name(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(PROBLEM_NAMES)}")
        return 1

    # Sanity-check against the study's stated minimums (does not block the
    # run -- just warns -- unless --quick was explicitly requested)
    if cfg.n_runs < 50 and not args.quick:
        print(f"[warning] n_runs = {cfg.n_runs} (the study specifies at least 50)")
    if cfg.max_iterations < 200 and not args.quick:
        print(f"[warning] max_iterations = {cfg.max_iterations} "
              f"(the study specifies at least 200)")

    print(f"MOBCO benchmark suite — {len(names)} problems, "
          f"{cfg.n_runs} runs x {cfg.max_iterations} iterations, "
          f"population {cfg.n_population} ({cfg.n_dogs} dogs), "
          f"archive {cfg.archive_size}")

    all_results = {}
    t_start = time.perf_counter()

    for idx, name in enumerate(names, 1):
        print(f"\n[{idx}/{len(names)}] {name}")
        problem = get_problem(name)
        res = run_problem(problem, cfg)
        all_results[name] = res

        xlsx = write_problem_workbook(res, results_dir)
        print(f"  workbook -> {xlsx}")

        if not args.no_figures:
            for f in make_all_figures(res, figures_dir):
                print(f"  figure   -> {f}")

        # Drop the (potentially large) per-generation archive snapshots once
        # figures/workbook are written, so memory usage stays flat across a
        # long multi-problem sweep instead of accumulating every problem's
        # full history in `all_results`.
        for out in res['run_outputs']:
            out['history']['snapshots'] = []

    master = write_master_summary(all_results, cfg, results_dir)
    print(f"\nmaster summary -> {master}")
    if not args.no_figures:
        print(f"overview figure -> {plot_overview(all_results, figures_dir)}")

    print(f"\ntotal wall-clock: {(time.perf_counter() - t_start) / 60:.1f} min")
    return 0


if __name__ == '__main__':
    sys.exit(main())
