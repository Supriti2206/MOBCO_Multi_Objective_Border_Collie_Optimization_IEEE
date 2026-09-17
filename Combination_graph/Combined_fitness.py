"""
combine_fitness.py

Purpose:
    You have a results folder for each algorithm
    (NSGA2_results, SPEA2_results, MOEAD_results, DSSEA_results, MOBCO_results),
    and each test function has one .xlsx file.

    In that file, the "Pareto_Solutions" sheet contains objective-wise fitness
    columns such as: f1_Objective_1, f2_Objective_2, f3_Objective_3, etc.
    (depending on the number of objectives for that test function).

    This script opens every xlsx file and FINDS all f1, f2, f3... columns
    (whether there are 2, 3, or 4 objectives - the count is detected automatically),
    COMBINES them into a single "Combined_Fitness" value for each solution/row,
    and adds a NEW SHEET ("Combined_Fitness") to that file.

Combination method (default):
    Objectives may have different scales (e.g. ZDT f1 may be in the [0,1] range,
    while EvoPINN losses may have a completely different scale). Therefore,
    taking a direct sum can be misleading because one objective may dominate
    the others. So:

        1. Each objective column is MIN-MAX NORMALIZED within that sheet
           [0, 1] range.
        2. An EQUAL-WEIGHT AVERAGE of all normalized objectives is taken
           to calculate Combined_Fitness.

    All objectives in this data are minimization objectives, so a lower
    Combined_Fitness indicates a better solution.

    If you want different weights (to give more importance to a particular
    objective), edit the WEIGHTS dict below.

    If you only want the raw sum (without normalization), set
    USE_NORMALIZATION = False.

Output:
    Default: original files are NOT modified - a new output folder is created
    (OUTPUT_DIR, default: "<INPUT_DIR>_with_combined_fitness") with the same
    folder structure and same filenames.

    The only difference is that each file contains an additional
    "Combined_Fitness" sheet.

    If you want the sheet to be added directly to the ORIGINAL files
    (in-place), set OUTPUT_DIR = INPUT_DIR below.

How to run:
    1. Set INPUT_DIR below to the path of your data folder
       (the folder containing the *_results folders).
    2. Run in the terminal: python combine_fitness.py
    3. All test-function files will be processed and progress will be printed.
"""

import os
import re
import glob
import openpyxl
from openpyxl.utils import get_column_letter

# CONFIG - change your settings here if needed

# Root folder containing sub-folders such as NSGA2_results, SPEA2_results,
# MOEAD_results, DSSEA_results, and MOBCO_results.
INPUT_DIR = "data"

# Output root folder. Default = INPUT_DIR + "_with_combined_fitness"
# (original files remain safe).
# To overwrite the original files in-place, use: OUTPUT_DIR = INPUT_DIR
OUTPUT_DIR = INPUT_DIR + "_with_combined_fitness"

# Sheet containing the objective columns (f1_, f2_, ...) to be combined.
SOURCE_SHEET = "Pareto_Solutions"

# Name of the new sheet that will be added.
OUTPUT_SHEET_NAME = "Combined_Fitness"

# True => normalize each objective to [0,1] using min-max normalization
# and then take the average (recommended because objectives may have
# different scales).
# False => directly calculate the raw sum
# (only use this when all objectives are already on the same scale).
USE_NORMALIZATION = True

# Custom weights (optional). None => all objectives receive equal weight (1/n).
# To specify custom weights, use a dictionary such as:
#   WEIGHTS = {0: 0.5, 1: 0.3, 2: 0.2}   # 0-based index, f1->0, f2->1, f3->2
# The weights do not need to sum to 1 - the script will normalize them automatically.
WEIGHTS = None

# CODE - generally no changes are required below

# Pattern for identifying column headers such as
# f1_something, f2_something, f3_something, etc.
F_COLUMN_PATTERN = re.compile(r"^f\d+(_.*)?$", re.IGNORECASE)


def find_objective_columns(header_row):
    """Find f1_, f2_, f3_... columns from the header row (list of cell values)
    and return (index, name) pairs sorted according to their numeric suffix
    (f1, f2, f3, ... f10)."""
    cols = []
    for idx, name in enumerate(header_row):
        if name is None:
            continue
        name_str = str(name).strip()
        if F_COLUMN_PATTERN.match(name_str):
            # Extract the number from f<number> so the columns are sorted
            # correctly (f1, f2, f3, ... f10).
            m = re.match(r"^f(\d+)", name_str, re.IGNORECASE)
            order_num = int(m.group(1)) if m else idx
            cols.append((order_num, idx, name_str))
    cols.sort(key=lambda t: t[0])
    return [(idx, name_str) for _, idx, name_str in cols]


def min_max_normalize(values):
    """Convert a list of floats to values normalized to [0, 1].
    If all values are the same (max == min), assign 0.0 to all values
    because there is no variation in this objective."""
    finite_vals = [v for v in values if v is not None]
    if not finite_vals:
        return [0.0] * len(values)
    vmin, vmax = min(finite_vals), max(finite_vals)
    span = vmax - vmin
    if span == 0:
        return [0.0 for _ in values]
    return [((v - vmin) / span) if v is not None else 0.0 for v in values]


def process_workbook(in_path, out_path):
    """Open an xlsx file, combine objectives from Pareto_Solutions,
    add a new sheet, and save the result to out_path."""

    # Load the workbook with data_only=True to read only VALUES
    # (formulas are read using their previously calculated values).
    # The main workbook (with formulas) will be loaded separately below
    # so that the remaining sheets and formatting stay untouched.
    values_wb = openpyxl.load_workbook(in_path, data_only=True)

    if SOURCE_SHEET not in values_wb.sheetnames:
        print(f"  [SKIP] '{SOURCE_SHEET}' sheet not found: {in_path}")
        return False

    ws_values = values_wb[SOURCE_SHEET]
    rows = list(ws_values.iter_rows(values_only=True))
    if not rows:
        print(f"  [SKIP] Sheet is empty: {in_path}")
        return False

    header = rows[0]
    data_rows = rows[1:]

    obj_cols = find_objective_columns(header)
    if not obj_cols:
        print(f"  [SKIP] No f1_/f2_/... objective column found: {in_path}")
        return False

    # Keep Solution / Best_Run columns for reference if they exist.
    solution_idx = header.index("Solution") if "Solution" in header else None
    best_run_idx = header.index("Best_Run") if "Best_Run" in header else None

    n_rows = len(data_rows)
    n_obj = len(obj_cols)

    # Extract raw values from each objective column.
    raw_matrix = []  # raw_matrix[obj_i] = list of raw values across rows
    for _, col_name in obj_cols:
        col_idx = header.index(col_name)
        raw_matrix.append([r[col_idx] for r in data_rows])

    # Normalize if enabled.
    if USE_NORMALIZATION:
        norm_matrix = [min_max_normalize(col) for col in raw_matrix]
    else:
        norm_matrix = raw_matrix

    # Determine weights.
    if WEIGHTS is None:
        weights = [1.0 / n_obj] * n_obj
    else:
        w = [WEIGHTS.get(i, 0.0) for i in range(n_obj)]
        total = sum(w) or 1.0
        weights = [x / total for x in w]

    # For each row, calculate combined fitness =
    # weighted average of the normalized objectives.
    combined = []
    raw_sum = []
    for r in range(n_rows):
        vals = [norm_matrix[o][r] for o in range(n_obj)]
        combined.append(sum(v * w for v, w in zip(vals, weights)))
        raw_sum.append(
            sum(
                raw_matrix[o][r]
                for o in range(n_obj)
                if raw_matrix[o][r] is not None
            )
        )

    # Prepare a new workbook copy (preserving formulas/formatting).
    # Load the original file again, this time without data_only.
    out_wb = openpyxl.load_workbook(in_path, data_only=False)

    if OUTPUT_SHEET_NAME in out_wb.sheetnames:
        del out_wb[OUTPUT_SHEET_NAME]
    ws_out = out_wb.create_sheet(OUTPUT_SHEET_NAME)

    # Create the header row:
    # Solution, Best_Run, raw f-columns, Combined_Fitness, Raw_Sum
    out_header = []
    if solution_idx is not None:
        out_header.append("Solution")
    if best_run_idx is not None:
        out_header.append("Best_Run")
    out_header += [name for _, name in obj_cols]
    out_header.append("Combined_Fitness")
    out_header.append("Raw_Sum")
    ws_out.append(out_header)

    for r in range(n_rows):
        row_out = []
        if solution_idx is not None:
            row_out.append(data_rows[r][solution_idx])
        if best_run_idx is not None:
            row_out.append(data_rows[r][best_run_idx])
        row_out += [raw_matrix[o][r] for o in range(n_obj)]
        row_out.append(combined[r])
        row_out.append(raw_sum[r])
        ws_out.append(row_out)

    # Set column widths for better readability.
    for i, _ in enumerate(out_header, start=1):
        ws_out.column_dimensions[get_column_letter(i)].width = 18

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_wb.save(out_path)
    print(f"  [OK] {n_rows} rows, {n_obj} objectives -> {out_path}")
    return True


def main():
    pattern = os.path.join(INPUT_DIR, "*_results", "*.xlsx")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No .xlsx files found for this pattern: {pattern}")
        print(
            "Is INPUT_DIR set correctly? Current INPUT_DIR =",
            os.path.abspath(INPUT_DIR)
        )
        return

    print(f"{len(files)} files found. Starting processing")
    ok_count = 0
    for in_path in files:
        rel = os.path.relpath(in_path, INPUT_DIR)
        out_path = os.path.join(OUTPUT_DIR, rel)
        print(f"Processing: {rel}")
        if process_workbook(in_path, out_path):
            ok_count += 1

    print(
        f"\nDone. Combined_Fitness sheet added to "
        f"{ok_count}/{len(files)} files."
    )
    print(f"Output folder: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()