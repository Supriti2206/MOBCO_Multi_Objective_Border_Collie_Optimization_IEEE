"""
Combined (5-algorithm) black-and-white figure generator for MOO benchmark results.

Algorithms: SPEA2, NSGA-II, MOBCO, MOEA/D, DSSEA
Test functions: 27 total (ZDT, DTLZ, Mixed_*, EvoPINN1-12)
Figure types (6): convergence, hypervolume, distribution, front, objective_space, decision_space

All figures are pure black & white (no colour used to separate algorithms).
Each algorithm gets ONE consistent visual identity, reused across every figure type:

    SPEA2    -> bold solid line   / filled circle marker      / solid  box edge (thick)
    MOEA/D   -> dashed line       / open square marker         / dashed box edge
    MOBCO    -> triangle markers  / filled triangle marker     / dash-dot box edge + triangle mean
    DSSEA    -> dotted line       / x marker                   / dotted box edge
    NSGA-II  -> light (thin) solid line / open diamond marker  / thin solid box edge

(Line-style is the literal spec you gave; for the scatter/box chart types -- where a
"line style" has no direct meaning -- that identity is extended into a matching
marker/hatch/edge-style so the same 5-way distinction still reads clearly in
grayscale. This is noted again in the accompanying message.)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Paths / constants

# Put ALL .xlsx files (all 5 algorithms x 27 test functions = 135 files) directly
# inside this one folder -- no per-algorithm subfolders needed. The lookup below
# also searches subfolders automatically, so it still works if you keep them
# organised into SPEA2_results/, NSGA2_results/, etc. instead.
DATA_ROOT = "data"
OUT_ROOT_BW = "output"
OUT_ROOT_COLOR = "output_color"

ALGOS = ["SPEA2", "NSGA2", "MOBCO", "MOEAD", "DSSEA"]

DISPLAY_NAME = {
    "SPEA2": "SPEA2",
    "NSGA2": "NSGA-II",
    "MOBCO": "MOBCO",
    "MOEAD": "MOEA/D",
    "DSSEA": "DSSEA",
}

# Distinct colour per algorithm, used only when MODE == "color". Line style /
# marker shape stay identical to the B&W version in both modes, so the colour
# figures still carry the shape distinction as a backup if ever printed in
# grayscale.
ALGO_COLOR = {
    "SPEA2": "#1f77b4",   # blue
    "MOEAD": "#d62728",   # red
    "MOBCO": "#2ca02c",   # green
    "DSSEA": "#9467bd",   # purple
    "NSGA2": "#ff7f0e",   # orange
}

# Current render mode: "bw" or "color". Set via set_mode() before plotting;
# generate_for_function() does this for you.
MODE = "bw"


def set_mode(mode):
    global MODE
    assert mode in ("bw", "color"), "mode must be 'bw' or 'color'"
    MODE = mode


def color_for(algo):
    return "black" if MODE == "bw" else ALGO_COLOR[algo]


def current_out_root():
    return OUT_ROOT_COLOR if MODE == "color" else OUT_ROOT_BW

# Style spec -- ONE definition per algorithm, reused everywhere

STYLE = {
    # zorder: thick/solid lines are drawn UNDERNEATH thin/dashed/dotted ones so the
    # latter never get visually buried when curves nearly coincide.
    # scatter_size values are matplotlib "s" (points^2) -- kept small & marker-edge
    # driven (mostly hollow) so overlapping point clouds read as distinguishable
    # shapes/texture rather than solid blobs.
    "SPEA2": dict(
        linestyle="-", linewidth=2.2, line_marker=None, markevery=None,
        scatter_marker="o", scatter_fill=True, scatter_size=9,
        hatch=None, box_linestyle="-", box_linewidth=2.2,
        zorder=2,
    ),
    "MOEAD": dict(
        linestyle=(0, (6, 3)), linewidth=1.5, line_marker=None, markevery=None,
        scatter_marker="s", scatter_fill=False, scatter_size=11,
        hatch="//", box_linestyle="--", box_linewidth=1.4,
        zorder=4,
    ),
    "MOBCO": dict(
        linestyle="None", linewidth=0.0, line_marker="^", markevery=0.05,
        scatter_marker="^", scatter_fill=True, scatter_size=12,
        hatch="\\\\", box_linestyle="-.", box_linewidth=1.4,
        zorder=6,
    ),
    "DSSEA": dict(
        linestyle=(0, (1, 1.6)), linewidth=1.8, line_marker=None, markevery=None,
        scatter_marker="x", scatter_fill=True, scatter_size=11,
        hatch="..", box_linestyle=":", box_linewidth=1.8,
        zorder=5,
    ),
    "NSGA2": dict(
        linestyle="-", linewidth=0.7, line_marker=None, markevery=None,
        scatter_marker="D", scatter_fill=False, scatter_size=7,
        hatch=None, box_linestyle="-", box_linewidth=0.8,
        zorder=3,
    ),
}

# Max points plotted per algorithm in any point-cloud chart (front / objective
# space / decision space / parallel coordinates). Subsampling is deterministic
# (fixed seed) so re-running the script reproduces identical figures.
MAX_SCATTER_POINTS = 90
MAX_PARCOORD_LINES = 35
RNG_SEED = 42

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 320,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
})


def subsample(*arrays, max_points=MAX_SCATTER_POINTS, seed=RNG_SEED):
    """Deterministically subsample parallel arrays down to max_points (no-op if smaller)."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return [a[idx] for a in arrays]


# Discovery -- searches DATA_ROOT recursively, so a flat folder or the original
# per-algorithm subfolders both work without any config changes.

def list_test_functions():
    pattern = os.path.join(DATA_ROOT, "**", "SPEA2_*.xlsx")
    files = sorted(glob.glob(pattern, recursive=True))
    fns = [os.path.basename(f)[len("SPEA2_"):-len(".xlsx")] for f in files]
    if not fns:
        print(
            f"WARNING: no 'SPEA2_*.xlsx' files found under '{os.path.abspath(DATA_ROOT)}'.\n"
            f"  -> Check that DATA_ROOT points at the folder holding your .xlsx files,\n"
            f"     and that filenames look like 'SPEA2_ZDT1.xlsx', 'NSGA2_ZDT1.xlsx', etc."
        )
    return fns


def xlsx_path(algo, fn):
    pattern = os.path.join(DATA_ROOT, "**", f"{algo}_{fn}.xlsx")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{algo}_{fn}.xlsx' anywhere under '{os.path.abspath(DATA_ROOT)}'. "
            f"Make sure that file exists (flat in the data folder, or in a subfolder)."
        )
    return matches[0]


# Data loading (cached per test function, all 5 algos, all needed sheets)

def load_all(fn):
    """Load every sheet needed for all 6 figure types, for all 5 algorithms."""
    data = {}
    for algo in ALGOS:
        path = xlsx_path(algo, fn)
        xl = pd.ExcelFile(path)
        d = {}
        d["iter_hist"] = pd.read_excel(xl, sheet_name="Iteration_History")
        d["run_summary"] = pd.read_excel(xl, sheet_name="Run_Summary")
        d["pareto"] = pd.read_excel(xl, sheet_name="Pareto_Solutions")
        d["obj_info"] = pd.read_excel(xl, sheet_name="Objective_Info")
        try:
            d["ref_front"] = pd.read_excel(xl, sheet_name="Reference_Front")
        except Exception:
            d["ref_front"] = None
        data[algo] = d
    return data


def n_objectives(data):
    return data["SPEA2"]["obj_info"].shape[0]


def obj_cols(data, algo="SPEA2"):
    info = data[algo]["obj_info"]
    return info["Objective_Name"].tolist()


def obj_label(data, i, algo="SPEA2"):
    """Short axis label f{i} - Name (minimise), 1-indexed."""
    info = data[algo]["obj_info"]
    row = info.iloc[i - 1]
    name = row["Objective_Name"].split("_", 1)[-1] if "_" in row["Objective_Name"] else row["Objective_Name"]
    direction = row.get("Direction", "Minimize")
    short = "min" if str(direction).lower().startswith("min") else "max"
    return f"f{i}: {name}\n({short})"


# Small style helpers

def draw_line(ax, x, y, algo, label=None):
    s = STYLE[algo]
    c = color_for(algo)
    lbl = label if label is not None else DISPLAY_NAME[algo]
    if s["line_marker"] is not None:
        ax.plot(
            x, y, color=c,
            linestyle="None",
            marker=s["line_marker"], markersize=4.2,
            markevery=s["markevery"],
            markerfacecolor=c, markeredgecolor=c,
            linewidth=s["linewidth"], label=lbl, zorder=s["zorder"],
        )
    else:
        ax.plot(
            x, y, color=c,
            linestyle=s["linestyle"], linewidth=s["linewidth"],
            label=lbl, zorder=s["zorder"],
        )


def draw_scatter(ax, x, y, algo, label=None, size_mult=1.0, alpha=0.85):
    s = STYLE[algo]
    c = color_for(algo)
    lbl = label if label is not None else DISPLAY_NAME[algo]
    fc = c if s["scatter_fill"] else "none"
    ax.scatter(
        x, y, marker=s["scatter_marker"], s=s["scatter_size"] * size_mult,
        facecolors=fc, edgecolors=c, linewidths=0.7,
        label=lbl, alpha=alpha, zorder=s["zorder"],
    )


def legend_once(fig_or_ax, ncol=5, loc="upper center", bbox=(0.5, -0.02)):
    handles, labels = fig_or_ax.get_legend_handles_labels() if hasattr(fig_or_ax, "get_legend_handles_labels") else ([], [])
    return handles, labels


def add_shared_legend(fig, ax, ncol=5, y=-0.04):
    handles, labels = ax.get_legend_handles_labels()
    # de-duplicate while preserving order
    seen = {}
    for h, l in zip(handles, labels):
        seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               bbox_to_anchor=(0.5, y), ncol=min(ncol, len(seen)), frameon=True)


def savefig(fig, fn, kind, outdir=None, use_tight=True):
    if outdir is None:
        outdir = current_out_root()
    d = os.path.join(outdir, kind)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{fn}_{kind}_combined.png")
    if use_tight:
        fig.savefig(path, bbox_inches="tight")
    else:
        fig.savefig(path)
    plt.close(fig)
    return path


# 1) CONVERGENCE  (4-panel: HV, IGD, GD, Front size vs iteration)

def plot_convergence(fn, data):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    panels = [
        ("Mean_Hypervolume", "Mean hypervolume (higher = better)"),
        ("Mean_IGD", "Mean IGD (lower = better)"),
        ("Mean_GD", "Mean GD (lower = better)"),
        ("Mean_Front_Size", "Mean front size"),
    ]
    for ax, (col, title) in zip(axes.flat, panels):
        for algo in ALGOS:
            df = data[algo]["iter_hist"]
            draw_line(ax, df["Iteration"], df[col], algo)
        ax.set_title(title)
        ax.set_xlabel("iteration")
    fig.suptitle(f"{fn} \u2014 convergence, all algorithms (mean over runs)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc="outside lower center", ncol=5)
    return savefig(fig, fn, "convergence", use_tight=False)


# 2) HYPERVOLUME  (single panel, mean hypervolume vs iteration)

def plot_hypervolume(fn, data):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for algo in ALGOS:
        df = data[algo]["iter_hist"]
        draw_line(ax, df["Iteration"], df["Mean_Hypervolume"], algo)
    ax.set_xlabel("iteration (generation)")
    ax.set_ylabel("hypervolume (normalised, higher = better)")
    ax.set_title(f"{fn} \u2014 hypervolume vs. generation, all algorithms")
    ax.legend(loc="lower right", ncol=1, markerscale=1.4)
    return savefig(fig, fn, "hypervolume")


# 3) DISTRIBUTION  (grouped box plots: 6 metric panels x 5 algorithms)

def plot_distribution(fn, data):
    metrics = ["Hypervolume", "IGD", "GD", "Spacing", "Spread", "Epsilon"]
    fig, axes = plt.subplots(1, 6, figsize=(20, 4.2))
    n = len(ALGOS)
    width = 0.7
    positions = np.arange(n)

    for ax, metric in zip(axes, metrics):
        values = [data[algo]["run_summary"][metric].values for algo in ALGOS]
        bp = ax.boxplot(
            values, positions=positions, widths=width, patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, markerfacecolor="none",
                             markeredgecolor="black"),
            medianprops=dict(color="black", linewidth=1.4),
        )
        for i, algo in enumerate(ALGOS):
            s = STYLE[algo]
            c = color_for(algo)
            box = bp["boxes"][i]
            if MODE == "bw":
                box.set_facecolor("white")
                box.set_edgecolor("black")
                if s["hatch"]:
                    box.set_hatch(s["hatch"])
            else:
                rgb = matplotlib.colors.to_rgb(c)
                box.set_facecolor((*rgb, 0.45))
                box.set_edgecolor(c)
            box.set_linestyle(s["box_linestyle"])
            box.set_linewidth(s["box_linewidth"])
        # whiskers/caps come in pairs per box
        for i, algo in enumerate(ALGOS):
            s = STYLE[algo]
            c = color_for(algo)
            for j in (2 * i, 2 * i + 1):
                bp["whiskers"][j].set_linestyle(s["box_linestyle"])
                bp["whiskers"][j].set_color(c)
                bp["caps"][j].set_color(c)
        # overlay a marker at the mean for each algorithm (shape matches its identity)
        for i, algo in enumerate(ALGOS):
            c = color_for(algo)
            mean_val = np.mean(data[algo]["run_summary"][metric].values)
            marker = STYLE[algo]["scatter_marker"]
            fc = c if STYLE[algo]["scatter_fill"] else "none"
            ax.scatter([positions[i]], [mean_val], marker=marker, s=22,
                       facecolors=fc, edgecolors=c, zorder=10, linewidths=0.8)

        ax.set_title(metric)
        ax.set_xticks(positions)
        ax.set_xticklabels([DISPLAY_NAME[a] for a in ALGOS], rotation=45, ha="right", fontsize=8.5)

    fig.suptitle(f"{fn} \u2014 indicator spread over independent runs, all algorithms", y=1.05)
    # build a manual legend describing box-edge style since boxplots don't auto-populate legend
    from matplotlib.patches import Patch
    if MODE == "bw":
        handles = [Patch(facecolor="white", edgecolor="black",
                          linestyle=STYLE[a]["box_linestyle"], linewidth=STYLE[a]["box_linewidth"],
                          hatch=STYLE[a]["hatch"], label=DISPLAY_NAME[a]) for a in ALGOS]
    else:
        handles = [Patch(facecolor=(*matplotlib.colors.to_rgb(color_for(a)), 0.45),
                          edgecolor=color_for(a),
                          linestyle=STYLE[a]["box_linestyle"], linewidth=STYLE[a]["box_linewidth"],
                          label=DISPLAY_NAME[a]) for a in ALGOS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=5)
    return savefig(fig, fn, "distribution")


# helpers shared by front / objective_space

def _best_run_points(data, algo, cols, max_points=MAX_SCATTER_POINTS):
    df = data[algo]["pareto"]
    arrays = [df[c].values for c in cols]
    return subsample(*arrays, max_points=max_points)


def plot_front(fn, data):
    """Reference front (grey) + each algorithm's best-run Pareto points, B&W markers."""
    nobj = n_objectives(data)
    cols = obj_cols(data, "SPEA2")

    if nobj == 2:
        fig, ax = plt.subplots(figsize=(8, 7))
        ref = data["SPEA2"]["ref_front"]
        if ref is not None:
            rx, ry = subsample(ref[cols[0]].values, ref[cols[1]].values, max_points=250)
            ax.scatter(rx, ry, s=5, c="0.8", label="Reference (expected)", zorder=1)
        for algo in ALGOS:
            x, y = _best_run_points(data, algo, cols)
            draw_scatter(ax, x, y, algo)
        ax.set_xlabel(obj_label(data, 1))
        ax.set_ylabel(obj_label(data, 2))
        ax.set_title(f"{fn} \u2014 Pareto front comparison (best run per algorithm)")
        ax.legend(loc="best", markerscale=1.7)
        return savefig(fig, fn, "front")

    elif nobj == 3:
        fig = plt.figure(figsize=(14.5, 7))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)
        ref = data["SPEA2"]["ref_front"]
        if ref is not None:
            rx, ry, rz = subsample(ref[cols[0]].values, ref[cols[1]].values, ref[cols[2]].values, max_points=250)
            ax1.scatter(rx, ry, rz, s=3, c="0.82", label="Reference", zorder=1)
            ax2.scatter(rx, rz, s=4, c="0.8", label="Reference", zorder=1)
        for algo in ALGOS:
            x, y, z = _best_run_points(data, algo, cols)
            s = STYLE[algo]
            c = color_for(algo)
            fc = c if s["scatter_fill"] else "none"
            ax1.scatter(x, y, z, marker=s["scatter_marker"], s=s["scatter_size"] * 0.7,
                        facecolors=fc, edgecolors=c, linewidths=0.6,
                        label=DISPLAY_NAME[algo], alpha=0.85, zorder=s["zorder"])
            draw_scatter(ax2, x, z, algo, size_mult=0.85)
        ax1.set_xlabel(obj_label(data, 1), labelpad=8)
        ax1.set_ylabel(obj_label(data, 2), labelpad=8)
        ax1.set_zlabel(obj_label(data, 3), labelpad=2)
        ax1.set_title("3-D view")
        ax2.set_xlabel(obj_label(data, 1))
        ax2.set_ylabel(obj_label(data, 3))
        ax2.set_title("f1 vs f3 projection")
        ax2.legend(loc="best", fontsize=8, ncol=2, markerscale=1.5)
        fig.suptitle(f"{fn} \u2014 Pareto front comparison (best run per algorithm)", y=1.02)
        return savefig(fig, fn, "front")

    else:  # 4 objectives -> parallel coordinates + log-log scatter of f1 vs f4
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7))
        ref = data["SPEA2"]["ref_front"]
        # normalise each objective 0-1 using reference (or pooled data) min/max
        all_vals = {c: [] for c in cols}
        if ref is not None:
            for c in cols:
                all_vals[c].append(ref[c].values)
        for algo in ALGOS:
            df = data[algo]["pareto"]
            for c in cols:
                all_vals[c].append(df[c].values)
        mins = {c: np.min(np.concatenate(all_vals[c])) for c in cols}
        maxs = {c: np.max(np.concatenate(all_vals[c])) for c in cols}

        def norm_row(vals):
            return [
                (v - mins[c]) / (maxs[c] - mins[c]) if maxs[c] > mins[c] else 0.5
                for v, c in zip(vals, cols)
            ]

        xs = np.arange(len(cols))
        if ref is not None:
            ref_sub = ref.sample(n=min(60, len(ref)), random_state=RNG_SEED)
            for _, row in ref_sub.iterrows():
                yv = norm_row([row[c] for c in cols])
                ax1.plot(xs, yv, color="0.82", linewidth=0.6, zorder=1)
        for algo in ALGOS:
            df = data[algo]["pareto"]
            df_sub = df.sample(n=min(MAX_PARCOORD_LINES, len(df)), random_state=RNG_SEED)
            s = STYLE[algo]
            c = color_for(algo)
            for _, row in df_sub.iterrows():
                yv = norm_row([row[c2] for c2 in cols])
                if s["line_marker"] is not None:
                    ax1.plot(xs, yv, color=c, linestyle="None", marker=s["line_marker"],
                              markersize=3.5, alpha=0.6, zorder=s["zorder"])
                else:
                    ax1.plot(xs, yv, color=c, linestyle=s["linestyle"],
                              linewidth=max(s["linewidth"] * 0.6, 0.5), alpha=0.55, zorder=s["zorder"])
        ax1.set_xticks(xs)
        ax1.set_xticklabels([obj_label(data, i + 1) for i in range(len(cols))], fontsize=8)
        ax1.set_ylabel("normalised objective value")
        ax1.set_title("parallel coordinates (all algorithms, grey = reference)")

        if ref is not None:
            rx, ry = subsample(ref[cols[0]].values, ref[cols[3]].values, max_points=150)
            ax2.scatter(rx, ry, s=5, c="0.8", label="Reference", zorder=1)
        for algo in ALGOS:
            x, y = _best_run_points(data, algo, [cols[0], cols[3]])
            draw_scatter(ax2, x, y, algo, size_mult=0.9)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(obj_label(data, 1))
        ax2.set_ylabel(obj_label(data, 4))
        ax2.set_title(f"f1 vs f4 (log-log)")
        ax2.legend(loc="best", fontsize=8, ncol=2, markerscale=1.5)

        # manual legend for parallel-coord panel
        from matplotlib.lines import Line2D
        handles = []
        for a in ALGOS:
            s = STYLE[a]
            c = color_for(a)
            if s["line_marker"]:
                handles.append(Line2D([0], [0], color=c, marker=s["line_marker"],
                                       linestyle="None", markersize=7, label=DISPLAY_NAME[a]))
            else:
                handles.append(Line2D([0], [0], color=c, linestyle=s["linestyle"],
                                       linewidth=s["linewidth"] + 0.6, label=DISPLAY_NAME[a]))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=5,
                   handlelength=3.6, markerscale=1.3)
        fig.suptitle(f"{fn} \u2014 Pareto front comparison (best run per algorithm)", y=1.03)
        return savefig(fig, fn, "front")


def plot_objective_space(fn, data):
    """Combined objective space achieved by each algorithm's best run (no reference layer)."""
    nobj = n_objectives(data)
    cols = obj_cols(data, "SPEA2")

    if nobj == 2:
        fig, ax = plt.subplots(figsize=(8, 7))
        for algo in ALGOS:
            x, y = _best_run_points(data, algo, cols)
            draw_scatter(ax, x, y, algo)
        ax.set_xlabel(obj_label(data, 1))
        ax.set_ylabel(obj_label(data, 2))
        ax.set_title(f"{fn} \u2014 objective space, all algorithms (best run)")
        ax.legend(loc="best", markerscale=1.7)
        return savefig(fig, fn, "objective_space")

    elif nobj == 3:
        fig = plt.figure(figsize=(14.5, 7))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)
        for algo in ALGOS:
            x, y, z = _best_run_points(data, algo, cols)
            s = STYLE[algo]
            c = color_for(algo)
            fc = c if s["scatter_fill"] else "none"
            ax1.scatter(x, y, z, marker=s["scatter_marker"], s=s["scatter_size"] * 0.7,
                        facecolors=fc, edgecolors=c, linewidths=0.6,
                        label=DISPLAY_NAME[algo], alpha=0.85, zorder=s["zorder"])
            draw_scatter(ax2, x, z, algo, size_mult=0.85)
        ax1.set_xlabel(obj_label(data, 1), labelpad=8)
        ax1.set_ylabel(obj_label(data, 2), labelpad=8)
        ax1.set_zlabel(obj_label(data, 3), labelpad=2)
        ax1.set_title("3-D view")
        ax2.set_xlabel(obj_label(data, 1))
        ax2.set_ylabel(obj_label(data, 3))
        ax2.set_title("f1 vs f3 projection")
        ax2.legend(loc="best", fontsize=8, ncol=2, markerscale=1.5)
        fig.suptitle(f"{fn} \u2014 objective space, all algorithms (best run)", y=1.02)
        return savefig(fig, fn, "objective_space")

    else:
        fig, ax = plt.subplots(figsize=(8.5, 7))
        for algo in ALGOS:
            x, y = _best_run_points(data, algo, [cols[0], cols[3]])
            draw_scatter(ax, x, y, algo, size_mult=0.95)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(obj_label(data, 1))
        ax.set_ylabel(obj_label(data, 4))
        ax.set_title(f"{fn} \u2014 objective space, all algorithms (best run, log-log)")
        ax.legend(loc="best", markerscale=1.6)
        return savefig(fig, fn, "objective_space")


# 6) DECISION SPACE  (x1 vs x2, all algorithms overlaid)

def plot_decision_space(fn, data):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for algo in ALGOS:
        df = data[algo]["pareto"]
        x, y = subsample(df["x1"].values, df["x2"].values)
        draw_scatter(ax, x, y, algo, size_mult=1.0, alpha=0.8)
    ax.set_xlabel("x1 (decision variable 1)")
    ax.set_ylabel("x2 (decision variable 2)")
    ax.set_title(f"{fn} \u2014 decision space, all algorithms (best run)")
    ax.legend(loc="best", markerscale=1.6)
    return savefig(fig, fn, "decision_space")


# Driver

PLOTTERS = {
    "convergence": plot_convergence,
    "hypervolume": plot_hypervolume,
    "distribution": plot_distribution,
    "front": plot_front,
    "objective_space": plot_objective_space,
    "decision_space": plot_decision_space,
}


def generate_for_function(fn, kinds=None, mode="bw", verbose=True):
    set_mode(mode)
    data = load_all(fn)
    kinds = kinds or list(PLOTTERS.keys())
    paths = []
    for kind in kinds:
        try:
            p = PLOTTERS[kind](fn, data)
            paths.append(p)
            if verbose:
                print(f"  [{fn}] ({mode}) {kind} -> {p}")
        except Exception as e:
            print(f"  [{fn}] ({mode}) {kind} FAILED: {e}")
    return paths


if __name__ == "__main__":
    import sys
    fns = list_test_functions()
    print(f"Found {len(fns)} test functions.")

    # Usage:
    #   python combined_plots_lib.py                -> all functions, both bw + color
    #   python combined_plots_lib.py ZDT1            -> just ZDT1, both bw + color
    #   python combined_plots_lib.py ZDT1 bw         -> just ZDT1, black & white only
    #   python combined_plots_lib.py ZDT1 color      -> just ZDT1, colour only
    #   python combined_plots_lib.py all color       -> every function, colour only
    args = sys.argv[1:]
    target = None
    modes = ["bw", "color"]
    if args:
        if args[0].lower() not in ("all",):
            target = args[0]
        if len(args) > 1 and args[1].lower() in ("bw", "color"):
            modes = [args[1].lower()]
        elif len(args) == 1 and args[0].lower() in ("bw", "color"):
            target = None
            modes = [args[0].lower()]

    targets = [target] if target else fns
    for mode in modes:
        print(f"\n=== Generating {mode.upper()} figures into '{OUT_ROOT_COLOR if mode == 'color' else OUT_ROOT_BW}/' ===")
        for fn in targets:
            generate_for_function(fn, mode=mode)