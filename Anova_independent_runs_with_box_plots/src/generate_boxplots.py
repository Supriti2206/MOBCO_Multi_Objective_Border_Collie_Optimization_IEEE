"""
generate_boxplots.py

Generates ONE box plot PER TEST PROBLEM (27 plots), each showing the
Combined_Fitness distribution (50 runs) for all 5 algorithms side by
side: this is the visual companion to the one-way ANOVA + Tukey HSD
results already computed by run_anova_combined_fitness.py.

Each plot's title shows the ANOVA F-statistic and p-value for that
problem. Algorithms are colored/labeled with a small letter code
derived from Tukey HSD ("compact letter display" style): algorithms
that share a letter are NOT significantly different from each other;
different letters ARE significantly different. This is the standard
way box plots communicate post-hoc results in a paper without needing
to draw every individual pairwise bracket.

Run AFTER compute_combined_fitness.py and run_anova_combined_fitness.py:
    python src/compute_combined_fitness.py
    python src/run_anova_combined_fitness.py
    python src/generate_boxplots.py

Output:
  results/boxplots/<Problem>.png   (27 PNG files, one per problem)
  results/boxplots/all_problems_grid.png  (all 27 in one grid, for a quick overview)
"""

from __future__ import annotations
import os
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT, "results")
BOXPLOT_DIR = os.path.join(RESULTS_DIR, "boxplots")
ALGO_ORDER = ["MOBCO", "SPEA2", "NSGA2", "MOEAD", "DSSEA"]
COLORS = {"MOBCO": "#e74c3c", "SPEA2": "#3498db", "NSGA2": "#2ecc71",
          "MOEAD": "#f39c12", "DSSEA": "#9b59b6"}


def compact_letter_display(algorithms: list[str], tukey_sub: pd.DataFrame, means: dict) -> dict:
    """
    Assigns a letter to each algorithm such that two algorithms share a
    letter if and only if Tukey HSD found them NOT significantly
    different. Standard greedy compact-letter-display construction,
    ordered by mean (best first).
    """
    ordered = sorted(algorithms, key=lambda a: -means[a])

    def is_sig(a, b):
        mask = (((tukey_sub["Group1"] == a) & (tukey_sub["Group2"] == b)) |
                ((tukey_sub["Group1"] == b) & (tukey_sub["Group2"] == a)))
        row = tukey_sub[mask]
        if row.empty:
            return False
        val = row.iloc[0]["Significant"]
        if isinstance(val, str):
            return val.strip().lower() in ("true", "reject")
        return bool(val)

    letters = {a: set() for a in ordered}
    groups: list[list[str]] = []
    for a in ordered:
        placed = False
        for g in groups:
            if all(not is_sig(a, b) for b in g):
                g.append(a)
                placed = True
                break
        if not placed:
            groups.append([a])
    for i, g in enumerate(groups):
        letter = string.ascii_lowercase[i]
        for a in g:
            letters[a].add(letter)
    return {a: "".join(sorted(letters[a])) for a in ordered}


def make_single_boxplot(ax, problem, fitness_df, anova_row, tukey_df):
    sub = fitness_df[fitness_df["Problem"] == problem]
    data = [sub.loc[sub["Algorithm"] == a, "Combined_Fitness"].to_numpy() for a in ALGO_ORDER]
    means = {a: sub.loc[sub["Algorithm"] == a, "Combined_Fitness"].mean() for a in ALGO_ORDER}

    bp = ax.boxplot(data, tick_labels=ALGO_ORDER, patch_artist=True, showmeans=True,
                     meanprops={"marker": "D", "markerfacecolor": "black", "markersize": 4})
    for patch, algo in zip(bp["boxes"], ALGO_ORDER):
        patch.set_facecolor(COLORS[algo])
        patch.set_alpha(0.6)

    decision = anova_row["decision"]
    p_val = anova_row["p_value"]
    p_str = f"{p_val:.3e}" if p_val > 0 else "< 1e-300"

    if decision == "Reject H0" and tukey_df is not None and not tukey_df.empty:
        tukey_sub = tukey_df[tukey_df["Problem"] == problem]
        cld = compact_letter_display(ALGO_ORDER, tukey_sub, means)
        ymax = max(v.max() for v in data if len(v))
        for i, algo in enumerate(ALGO_ORDER, start=1):
            ax.text(i, ymax * 1.03, cld[algo], ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_title(f"{problem}\nF={anova_row['F_statistic']:.2f}, p={p_str}, {decision}",
                 fontsize=9)
    ax.set_ylabel("Combined Fitness (0=worst, 1=best)", fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)


def main():
    os.makedirs(BOXPLOT_DIR, exist_ok=True)
    fitness_df = pd.read_csv(os.path.join(RESULTS_DIR, "Combined_Fitness_per_Run.csv"))
    anova_df = pd.read_csv(os.path.join(RESULTS_DIR, "ANOVA_Combined_Fitness_one_pvalue_per_problem.csv"))
    tukey_path = os.path.join(RESULTS_DIR, "ANOVA_Combined_Fitness_Tukey_PostHoc.csv")
    tukey_df = pd.read_csv(tukey_path) if os.path.exists(tukey_path) else None

    problems = sorted(fitness_df["Problem"].unique())

    # Individual plot per problem
    for problem in problems:
        anova_row = anova_df[anova_df["Problem"] == problem].iloc[0]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        make_single_boxplot(ax, problem, fitness_df, anova_row, tukey_df)
        fig.tight_layout()
        out_path = os.path.join(BOXPLOT_DIR, f"{problem}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"Saved {len(problems)} individual box plots to: {BOXPLOT_DIR}")

    # Overview grid: all 27 problems in one image (5 cols x 6 rows fits 27)
    n_cols, n_rows = 5, int(np.ceil(len(problems) / 5))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4))
    axes = axes.flatten()
    for i, problem in enumerate(problems):
        anova_row = anova_df[anova_df["Problem"] == problem].iloc[0]
        make_single_boxplot(axes[i], problem, fitness_df, anova_row, tukey_df)
    for j in range(len(problems), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    grid_path = os.path.join(BOXPLOT_DIR, "all_problems_grid.png")
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)
    print(f"Saved overview grid to: {grid_path}")


if __name__ == "__main__":
    main()
