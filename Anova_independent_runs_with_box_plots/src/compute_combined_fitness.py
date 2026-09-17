"""
compute_combined_fitness.py

Folds all 7 performance metrics (Hypervolume, IGD, GD, Spacing, Spread,
Epsilon, Runtime) into ONE composite fitness number per run, giving
50 independent composite values per algorithm per problem (same shape
as a single raw metric like Hypervolume) -- exactly what's needed to
run a plain one-way ANOVA with one p-value per problem.

METHOD (min-max normalization, direction-aware, then equal-weighted average):

For each Problem, across all 5 algorithms x 50 runs = 250 values per metric:
  1. For "higher is better" metrics (Hypervolume):
         normalized = (value - min) / (max - min)
  2. For "lower is better" metrics (IGD, GD, Spacing, Spread, Epsilon, Runtime):
         normalized = (max - value) / (max - min)
  After this, for every metric: 1.0 = best observed, 0.0 = worst observed,
  on the same 0-1 scale, so no single metric (e.g. Runtime in seconds vs.
  IGD in tiny fractions) dominates just because of its raw units.

  3. Combined_Fitness (per run) = mean of the 7 normalized values (equal weights).
     Higher Combined_Fitness = better overall run.

Edge case: if a metric is exactly constant across all 250 values for a
problem (max == min, no discriminating information), its normalized
value is set to 1.0 for every run rather than dividing by zero -- this
means that metric contributes no penalty/reward for that problem, since
there's nothing to distinguish on it.

Normalization is done PER PROBLEM (not globally across all 27 problems),
because different problems have very different natural scales for the
same metric (e.g. Runtime on ZDT1 vs. Runtime on an EvoPINN problem) --
mixing those scales globally would make cross-problem comparisons
meaningless.

Output:
  results/Combined_Fitness_per_Run.csv   (Problem, Algorithm, Run, Combined_Fitness)
  results/Combined_Fitness_per_Run.xlsx

Run from the project root:
    python src/compute_combined_fitness.py
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from load_data import build_long_dataframe, METRICS, METRIC_DIRECTION

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_ROOT = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")


def normalize_metric(values: np.ndarray, direction: str) -> np.ndarray:
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        # No discriminating information for this metric on this problem.
        return np.ones_like(values, dtype=float)
    if direction == "higher":
        return (values - vmin) / (vmax - vmin)
    else:  # "lower" is better
        return (vmax - values) / (vmax - vmin)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    long_df, algorithms, problems, load_errors = build_long_dataframe(DATA_ROOT)
    print(f"Loaded {len(long_df)} rows | {len(algorithms)} algorithms | {len(problems)} problems")

    out_rows = []
    for problem in problems:
        sub = long_df[long_df["Problem"] == problem].copy()

        normalized_cols = {}
        for metric in METRICS:
            direction = METRIC_DIRECTION[metric]
            normalized_cols[metric] = normalize_metric(sub[metric].to_numpy(dtype=float), direction)

        norm_df = pd.DataFrame(normalized_cols)
        combined_fitness = norm_df.mean(axis=1).to_numpy()

        result = sub[["Problem", "Algorithm", "Run"]].copy()
        result["Combined_Fitness"] = combined_fitness
        out_rows.append(result)

    out_df = pd.concat(out_rows, ignore_index=True)

    csv_path = os.path.join(RESULTS_DIR, "Combined_Fitness_per_Run.csv")
    xlsx_path = os.path.join(RESULTS_DIR, "Combined_Fitness_per_Run.xlsx")
    out_df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Combined_Fitness_per_Run", index=False)

    print(f"\n{len(out_df)} rows written (should be {len(problems)} problems x "
          f"{len(algorithms)} algorithms x 50 runs = {len(problems)*len(algorithms)*50}).")
    print(f"Written to: {csv_path}")
    print(f"Written to: {xlsx_path}")

    # Quick sanity check print: mean Combined_Fitness per algorithm, ZDT1
    sample = out_df[out_df["Problem"] == "ZDT1"]
    print("\nSanity check -- mean Combined_Fitness per algorithm, ZDT1:")
    print(sample.groupby("Algorithm")["Combined_Fitness"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()
