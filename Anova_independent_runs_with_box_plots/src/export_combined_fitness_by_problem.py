"""
export_combined_fitness_by_problem.py

Takes results/Combined_Fitness_per_Run.csv (produced by
compute_combined_fitness.py) and writes it out as ONE Excel workbook
with a SEPARATE SHEET PER TEST PROBLEM (27 sheets), each sheet laid
out wide -- one column per algorithm, 50 rows (one per run) -- so it's
ready to eyeball, paste into a stats calculator, or hand to a mentor
one problem at a time.

Each sheet's columns: Run, MOBCO, SPEA2, NSGA2, MOEAD, DSSEA
(all 5 algorithm columns, direction-independent -- higher
Combined_Fitness is always better, since it's already normalized).

Run AFTER compute_combined_fitness.py:
    python src/compute_combined_fitness.py
    python src/export_combined_fitness_by_problem.py

Output:
  results/Combined_Fitness_by_Problem.xlsx   (27 sheets, one per problem)
"""

from __future__ import annotations
import os
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT, "results")


def main():
    in_path = os.path.join(RESULTS_DIR, "Combined_Fitness_per_Run.csv")
    df = pd.read_csv(in_path)

    problems = sorted(df["Problem"].unique())
    algorithms = sorted(df["Algorithm"].unique())

    out_path = os.path.join(RESULTS_DIR, "Combined_Fitness_by_Problem.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for problem in problems:
            sub = df[df["Problem"] == problem]

            # Wide format: one column per algorithm, aligned by Run number.
            wide = sub.pivot(index="Run", columns="Algorithm", values="Combined_Fitness")
            wide = wide.reindex(columns=algorithms)  # consistent column order every sheet
            wide = wide.reset_index()

            # Excel sheet names must be <= 31 chars; all 27 problem names
            # here are already under that limit, but truncate defensively
            # in case new problems are added later.
            sheet_name = problem[:31]
            wide.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Wrote {len(problems)} sheets (one per problem) to: {out_path}")
    print("Sheets:", problems)


if __name__ == "__main__":
    main()
