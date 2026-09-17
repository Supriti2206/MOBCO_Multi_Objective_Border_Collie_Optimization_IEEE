"""
load_data.py

Loads per-run performance metric data (Hypervolume, IGD, GD, Spacing, Spread,
Epsilon, Runtime) from the 'Run_Summary' sheet of every algorithm/problem
result workbook and assembles it into a single tidy (long-format) DataFrame.
"""

from __future__ import annotations
import os
import glob
import openpyxl
import pandas as pd

METRICS = ["Hypervolume", "IGD", "GD", "Spacing", "Spread", "Epsilon", "Runtime"]

METRIC_DIRECTION = {
    "Hypervolume": "higher",
    "IGD": "lower",
    "GD": "lower",
    "Spacing": "lower",
    "Spread": "lower",
    "Epsilon": "lower",
    "Runtime": "lower",
}


def discover_algorithms(data_root: str) -> list[str]:
    algos = []
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and entry.endswith("_results"):
            algos.append(entry[: -len("_results")])
    return algos


def discover_problems(data_root: str, algorithms: list[str]) -> list[str]:
    problem_sets = []
    for algo in algorithms:
        folder = os.path.join(data_root, f"{algo}_results")
        names = set()
        for path in glob.glob(os.path.join(folder, f"{algo}_*.xlsx")):
            fname = os.path.basename(path)[:-len(".xlsx")]
            problem = fname[len(algo) + 1:]
            if problem == "Master_Summary":
                continue
            names.add(problem)
        problem_sets.append(names)

    common = set.intersection(*problem_sets) if problem_sets else set()
    all_seen = set.union(*problem_sets) if problem_sets else set()
    missing = all_seen - common
    if missing:
        print(f"[WARNING] These problems are NOT present for all algorithms "
              f"and will be EXCLUDED from analysis: {sorted(missing)}")
    return sorted(common)


def load_run_summary(path: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, data_only=True)
    if "Run_Summary" not in wb.sheetnames:
        raise ValueError(f"'Run_Summary' sheet not found in {path}")
    ws = wb["Run_Summary"]
    rows = list(ws.iter_rows(values_only=True))
    header, data_rows = rows[0], rows[1:]
    df = pd.DataFrame(data_rows, columns=header)
    return df


def build_long_dataframe(data_root: str) -> pd.DataFrame:
    algorithms = discover_algorithms(data_root)
    problems = discover_problems(data_root, algorithms)

    frames = []
    load_errors = []
    for problem in problems:
        for algo in algorithms:
            path = os.path.join(data_root, f"{algo}_results", f"{algo}_{problem}.xlsx")
            try:
                df = load_run_summary(path)
            except Exception as e:
                load_errors.append(f"{path}: {e}")
                continue

            missing_cols = [m for m in METRICS if m not in df.columns]
            if missing_cols:
                load_errors.append(f"{path}: missing metric columns {missing_cols}")
                continue

            n_before = len(df)
            df = df.dropna(subset=METRICS)
            n_after = len(df)
            if n_after != n_before:
                load_errors.append(
                    f"{path}: dropped {n_before - n_after} row(s) with NaN metric values"
                )

            df = df[["Run"] + METRICS].copy()
            df.insert(0, "Algorithm", algo)
            df.insert(0, "Problem", problem)
            frames.append(df)

    if load_errors:
        print("[DATA QUALITY NOTES]")
        for e in load_errors:
            print(" -", e)

    if not frames:
        raise RuntimeError("No data loaded — check data_root path and file layout.")

    long_df = pd.concat(frames, ignore_index=True)
    for m in METRICS:
        long_df[m] = pd.to_numeric(long_df[m], errors="coerce")

    return long_df, algorithms, problems, load_errors


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..", "data")
    df, algos, probs, errs = build_long_dataframe(root)
    print(f"\nLoaded {len(df)} rows | {len(algos)} algorithms | {len(probs)} problems")
