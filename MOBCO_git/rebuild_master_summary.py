"""
rebuild_master_summary.py -- Standalone recovery tool: reconstructs
MOBCO_Master_Summary.xlsx by re-reading every per-problem
MOBCO_<Name>.xlsx workbook already sitting in results/, instead of
re-running the experiments. Useful if the master summary got deleted/
corrupted but the per-problem workbooks are still intact.

*** CAUTION -- likely broken against the current pipeline ***
This script reads 'IGD+' from the Run_Summary and Statistics sheets
(row['IGD+'], stat('IGD+', ...)), but excel_writer.py's
write_problem_workbook() does not currently write any 'IGD+' column --
only Hypervolume, IGD, GD, Spacing, Spread, Epsilon, Runtime. Running this
against workbooks produced by the current excel_writer.py will raise a
KeyError on row['IGD+']. Either an 'IGD+' (IGD-plus) indicator existed in
an earlier version of excel_writer.py and was later removed, or this
script predates a column rename -- worth reconciling the two before
relying on it.
"""

import glob
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / 'results'


def family_of(name):
    """Classify a problem name into its benchmark family, by name prefix,
    for the 'Family' column (mirrors the same logic in excel_writer.py's
    write_master_summary)."""
    if name.startswith('EvoPINN'):
        return 'EvoPINN'
    if name.startswith('Mixed'):
        return 'Mixed'
    if name.startswith('ZDT'):
        return 'ZDT'
    return 'DTLZ'


def main():
    """Scan results/ for every MOBCO_<Name>.xlsx (excluding the master
    summary itself), re-extract each one's Run_Summary/Statistics/
    Parameters sheets, and rebuild the same 3-sheet Master_Summary workbook
    that write_master_summary() would have produced directly from a live
    run."""
    files = sorted(f for f in glob.glob(str(RESULTS_DIR / 'MOBCO_*.xlsx'))
                    if 'Master_Summary' not in f)
    if not files:
        print('no per-problem workbooks found in results/')
        return

    summary_rows, long_rows, config_rows = [], [], None
    for f in files:
        name = Path(f).stem.replace('MOBCO_', '')
        run_summary = pd.read_excel(f, sheet_name='Run_Summary')
        stats = pd.read_excel(f, sheet_name='Statistics').set_index('Metric')
        params = pd.read_excel(f, sheet_name='Parameters').set_index('Parameter')['Value']

        for _, row in run_summary.iterrows():
            long_rows.append({
                'Problem': name, 'Run': int(row['Run']),
                'Hypervolume': row['Hypervolume'], 'IGD': row['IGD'],
                'IGD+': row['IGD+'], 'GD': row['GD'],
                'Spacing': row['Spacing'], 'Spread': row['Spread'],
                'Epsilon': row['Epsilon'], 'Runtime': row['Runtime'],
            })

        def stat(metric, col):
            """Look up one (metric, stat-column) cell from this workbook's
            Statistics sheet, returning NaN if the metric row is missing
            (keeps this script tolerant of older/newer workbook schemas
            for everything except the required 'IGD+' column above)."""
            return float(stats.loc[metric, col]) if metric in stats.index else float('nan')

        summary_rows.append({
            'Problem': name, 'Family': family_of(name),
            'Dim': int(params['Decision Variables (dim)']),
            'N_Objectives': int(params['Number of Objectives']),
            'Directions': params['Objective Directions'],
            'Reference_Front': params['Reference Front'],
            'HV_Mean': stat('Hypervolume', 'Mean'), 'HV_Std': stat('Hypervolume', 'Std'),
            'HV_Best': stat('Hypervolume', 'Best'),
            'IGD_Mean': stat('IGD', 'Mean'), 'IGD_Std': stat('IGD', 'Std'),
            'IGDPlus_Mean': stat('IGD+', 'Mean'), 'IGDPlus_Std': stat('IGD+', 'Std'),
            'GD_Mean': stat('GD', 'Mean'), 'Spacing_Mean': stat('Spacing', 'Mean'),
            'Spread_Mean': stat('Spread', 'Mean'), 'Epsilon_Mean': stat('Epsilon', 'Mean'),
            'Front_Size_Mean': stat('Front_Size', 'Mean'),
            'Runtime_Mean_sec': stat('Runtime', 'Mean'),
            'Runtime_Total_sec': float(run_summary['Runtime'].sum()),
            'Runs': int(params['Independent Runs']),
            'Iterations': int(params['Max Iterations']),
        })

        if config_rows is None:
            # First workbook processed becomes the template for the shared
            # Configuration sheet (assumes every problem in the sweep used
            # the same config -- true for a normal single-experiment run).
            config_rows = [{'Parameter': k, 'Value': str(v)} for k, v in params.items()]

    out = RESULTS_DIR / 'MOBCO_Master_Summary.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Master_Summary', index=False)
        pd.DataFrame(long_rows).to_excel(writer, sheet_name='All_Runs', index=False)
        pd.DataFrame(config_rows).to_excel(writer, sheet_name='Configuration', index=False)

    print(f'rebuilt {out} from {len(files)} problem workbooks '
          f'({len(long_rows)} total runs)')


if __name__ == '__main__':
    main()
