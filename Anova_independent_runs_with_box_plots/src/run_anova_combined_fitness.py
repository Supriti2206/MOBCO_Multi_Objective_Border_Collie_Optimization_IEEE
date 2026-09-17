"""
run_anova_combined_fitness.py

Runs a plain One-Way ANOVA (5 algorithm groups, n=50 each) on the
Combined_Fitness score computed by compute_combined_fitness.py.
Because Combined_Fitness is already a single number per run (not 7
separate metrics), this gives exactly ONE p-value per problem directly.

Also runs Tukey HSD post-hoc where the ANOVA is significant.

Run AFTER compute_combined_fitness.py:
    python src/compute_combined_fitness.py
    python src/export_combined_fitness_by_problem.py
    python src/run_anova_combined_fitness.py

Output:
  results/ANOVA_Combined_Fitness_one_pvalue_per_problem.csv / .xlsx
  results/ANOVA_Combined_Fitness_Tukey_PostHoc.csv
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT, "results")
ALPHA = 0.05


def eta_squared(groups: dict) -> float:
    all_vals = np.concatenate(list(groups.values()))
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())
    return ss_between / ss_total if ss_total > 0 else np.nan


def main():
    fitness_path = os.path.join(RESULTS_DIR, "Combined_Fitness_per_Run.csv")
    df = pd.read_csv(fitness_path)

    problems = sorted(df["Problem"].unique())
    algorithms = sorted(df["Algorithm"].unique())

    anova_rows = []
    tukey_rows = []

    for problem in problems:
        sub = df[df["Problem"] == problem]
        groups = {a: sub.loc[sub["Algorithm"] == a, "Combined_Fitness"].to_numpy(float)
                  for a in algorithms}

        F, p = stats.f_oneway(*groups.values())
        k = len(groups)
        n = sum(len(v) for v in groups.values())
        df1, df2 = k - 1, n - k
        eta2 = eta_squared(groups)
        decision = "Reject H0" if p < ALPHA else "Fail to Reject H0"

        anova_rows.append({
            "Problem": problem, "test_used": "One-Way ANOVA (Combined Fitness)",
            "F_statistic": F, "df1": df1, "df2": df2, "p_value": p,
            "eta_squared": eta2, "decision": decision,
        })

        if decision == "Reject H0":
            values, labels = [], []
            for a, vals in groups.items():
                values.extend(vals)
                labels.extend([a] * len(vals))
            tuk = pairwise_tukeyhsd(np.array(values), np.array(labels), alpha=ALPHA)
            tdf = pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0])
            tdf = tdf.rename(columns={"group1": "Group1", "group2": "Group2",
                                       "meandiff": "Mean_Diff", "p-adj": "p_adj",
                                       "reject": "Significant"})
            tdf.insert(0, "Problem", problem)
            tukey_rows.append(tdf)

    anova_df = pd.DataFrame(anova_rows)
    tukey_df = pd.concat(tukey_rows, ignore_index=True) if tukey_rows else pd.DataFrame()

    csv_path = os.path.join(RESULTS_DIR, "ANOVA_Combined_Fitness_one_pvalue_per_problem.csv")
    xlsx_path = os.path.join(RESULTS_DIR, "ANOVA_Combined_Fitness_one_pvalue_per_problem.xlsx")
    tukey_path = os.path.join(RESULTS_DIR, "ANOVA_Combined_Fitness_Tukey_PostHoc.csv")

    anova_df.to_csv(csv_path, index=False)
    tukey_df.to_csv(tukey_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        anova_df.to_excel(writer, sheet_name="ANOVA_p_per_problem", index=False)
        tukey_df.to_excel(writer, sheet_name="Tukey_PostHoc", index=False)

    n_sig = int((anova_df["decision"] == "Reject H0").sum())
    print(f"{len(anova_df)} problems tested -> exactly 1 p-value each.")
    print(f"Significant at alpha={ALPHA}: {n_sig} / {len(anova_df)}")
    print(f"Written to: {csv_path}")
    print(f"Written to: {xlsx_path}")
    print(anova_df[["Problem", "F_statistic", "p_value", "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
