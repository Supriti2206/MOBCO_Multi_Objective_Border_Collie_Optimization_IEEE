"""
visualization.py -- Matplotlib figure generation for the IEEE paper: for
each problem, produces an "expected vs obtained" front plot (shape depends
on n_obj), a hypervolume-vs-generation convergence curve, objective-space
and decision-space scatter plots across all runs, a 4-panel convergence
summary, and per-run indicator boxplots -- plus one cross-problem overview
bar chart. Uses the non-interactive 'Agg' backend since these are always
saved to disk, never displayed live.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

EXPECTED_STYLE = dict(color='#b0b0b0', s=12, alpha=0.85, label='Expected (reference front)')
OBTAINED_STYLE = dict(color='#c0392b', s=22, alpha=0.9, label='Reality (MOBCO best run)')
COMBINED_STYLE = dict(color='#2471a3', s=10, alpha=0.45, label='Reality (all runs combined)')


def _labels(problem):
    """Build per-axis labels like 'f1 - Cost\n(minimise)' for plotting,
    using the problem's optional 'objective_names' where available."""
    names = problem.get('objective_names') or []
    out = []
    for i, t in enumerate(problem['objective_types']):
        base = names[i] if i < len(names) else f'f{i + 1}'
        arrow = 'minimise' if t == 'min' else 'MAXIMISE'
        out.append(f'f{i + 1} · {base}\n({arrow})')
    return out


def _save(fig, path):
    """Tight-layout, save at 130 dpi, close the figure (to free memory
    across a long multi-problem/multi-figure sweep), and return the path
    as a string."""
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_front(result, fig_dir):
    """
    'Expected vs reality' Pareto front comparison, with the layout adapted
    to the number of objectives:
      - 2 objectives: single 2D scatter (reference / combined / best run)
      - 3 objectives: a 3D scatter view + an f1-vs-f3 2D projection
      - 4+ objectives (EvoPINN): parallel-coordinates plot (all objectives
        at once, normalized to [0,1]) + a log-log f1-vs-f4 scatter, since a
        direct 4D+ scatter isn't visualizable
    """
    problem = result['problem']
    R = result['reference_front']
    best = result['run_outputs'][result['best_run']]['front']
    comb = result['combined_front']
    lab = _labels(problem)
    m = problem['n_obj']
    path = Path(fig_dir) / f"{problem['name']}_front.png"

    if m == 2:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        ax.scatter(R[:, 0], R[:, 1], **EXPECTED_STYLE)
        ax.scatter(comb[:, 0], comb[:, 1], **COMBINED_STYLE)
        ax.scatter(best[:, 0], best[:, 1], **OBTAINED_STYLE)
        ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1])
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_title(f"{problem['name']} — expected vs. reality")
        return _save(fig, path)

    if m == 3:
        fig = plt.figure(figsize=(12.5, 5.6))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(R[:, 0], R[:, 1], R[:, 2], color='#b0b0b0', s=8, alpha=0.6,
                   label='Expected')
        ax.scatter(best[:, 0], best[:, 1], best[:, 2], color='#c0392b', s=18,
                   label='MOBCO best run')
        ax.set_xlabel(lab[0].replace('\n', ' '), fontsize=7)
        ax.set_ylabel(lab[1].replace('\n', ' '), fontsize=7)
        ax.set_zlabel(lab[2].replace('\n', ' '), fontsize=7)
        ax.legend(fontsize=8)
        ax.set_title(f"{problem['name']} — 3-D view")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(R[:, 0], R[:, 2], **EXPECTED_STYLE)
        ax2.scatter(comb[:, 0], comb[:, 2], **COMBINED_STYLE)
        ax2.scatter(best[:, 0], best[:, 2], **OBTAINED_STYLE)
        ax2.set_xlabel(lab[0]); ax2.set_ylabel(lab[2])
        ax2.grid(alpha=0.25); ax2.legend(fontsize=8)
        ax2.set_title('f1 vs f3 projection')
        return _save(fig, path)

    # 4+ objectives (EvoPINN): parallel coordinates + scatter matrix
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1)

    def _norm(A, lo, hi):
        """Min-max normalize columns of A into [0,1] using shared lo/hi
        (computed across R+comb+best together, so all three series share
        the same scale on the parallel-coordinates plot)."""
        span = np.where(hi - lo <= 0, 1.0, hi - lo)
        return (A - lo) / span

    stack = np.vstack([R, comb, best]) if len(comb) else np.vstack([R, best])
    lo, hi = stack.min(axis=0), stack.max(axis=0)
    xs = np.arange(m)
    for row in _norm(R, lo, hi):
        ax.plot(xs, row, color='#b0b0b0', alpha=0.25, lw=0.7)
    for row in _norm(best, lo, hi):
        ax.plot(xs, row, color='#c0392b', alpha=0.55, lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([l.replace('\n', '\n') for l in lab], fontsize=6.5)
    ax.set_ylabel('normalised objective value')
    ax.set_title(f"{problem['name']} — parallel coordinates\n"
                 f"grey = expected (reference), red = MOBCO best run", fontsize=9)
    ax.grid(alpha=0.2)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(R[:, 0], R[:, 3], **EXPECTED_STYLE)
    ax2.scatter(best[:, 0], best[:, 3], **OBTAINED_STYLE)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel(lab[0]); ax2.set_ylabel(lab[3])
    ax2.grid(alpha=0.25); ax2.legend(fontsize=8)
    ax2.set_title('residual loss vs data loss (log-log)', fontsize=9)
    return _save(fig, path)


def plot_hypervolume(result, fig_dir):
    """Generation-wise hypervolume: mean curve +/- 1 std band across the
    traced runs (config.history_metric_runs runs record a snapshot every
    generation). Standalone companion to the 4-panel convergence figure."""
    problem = result['problem']
    h = result['history']
    path = Path(fig_dir) / f"{problem['name']}_hypervolume.png"
    if h is None:
        return None
    it = h['iteration']
    mean = h['hv']
    std = h.get('hv_std', np.zeros_like(mean))
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(it, mean, color='#1e8449', lw=2.0, label='Mean hypervolume')
    ax.fill_between(it, mean - std, mean + std, color='#1e8449', alpha=0.22,
                    label=f"+/- 1 std ({h['n_traced_runs']} runs)")
    ax.set_xlabel('iteration (generation)')
    ax.set_ylabel('hypervolume (normalised, higher = better)')
    ax.set_title(f"{problem['name']} — hypervolume vs. generation")
    ax.grid(alpha=0.25)
    ax.legend(loc='best', fontsize=8)
    return _save(fig, path)


def plot_objective_space(result, fig_dir):
    """Every run's final Pareto front overlaid in objective space
    (f1 vs f2, colour-coded by run index), plus the reference front in the
    background for context."""
    problem = result['problem']
    R = result['reference_front']
    outs = result['run_outputs']
    lab = _labels(problem)
    path = Path(fig_dir) / f"{problem['name']}_objective_space.png"

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.scatter(R[:, 0], R[:, 1], color='#b0b0b0', s=10, alpha=0.5,
              label='Expected (reference front)', zorder=1)
    cmap = plt.get_cmap('viridis')
    n_runs = len(outs)
    for i, out in enumerate(outs):
        F = out['front']
        if len(F) == 0:
            continue
        ax.scatter(F[:, 0], F[:, 1], color=cmap(i / max(1, n_runs - 1)),
                  s=10, alpha=0.6, zorder=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, n_runs))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('run index')
    ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1])
    ax.set_title(f"{problem['name']} — objective space (all {n_runs} runs)")
    ax.grid(alpha=0.25)
    ax.legend(loc='best', fontsize=8)
    return _save(fig, path)


def plot_decision_space(result, fig_dir):
    """Every run's final Pareto-optimal decisions in decision space (first
    two decision variables), colour-coded by run index."""
    problem = result['problem']
    outs = result['run_outputs']
    dim = problem['dim']
    path = Path(fig_dir) / f"{problem['name']}_decision_space.png"

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    cmap = plt.get_cmap('plasma')
    n_runs = len(outs)
    for i, out in enumerate(outs):
        X = out['front_decisions']
        if len(X) == 0:
            continue
        x1 = X[:, 0]
        x2 = X[:, 1] if dim > 1 else np.zeros(len(X))
        ax.scatter(x1, x2, color=cmap(i / max(1, n_runs - 1)),
                  s=10, alpha=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, n_runs))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('run index')
    ax.set_xlabel('x1 (decision variable 1)')
    ax.set_ylabel('x2 (decision variable 2)' if dim > 1 else '(constant, dim = 1)')
    ax.set_title(f"{problem['name']} — decision space (all {n_runs} runs, "
                f"dim={dim})")
    ax.grid(alpha=0.25)
    return _save(fig, path)


def plot_convergence(result, fig_dir):
    """4-panel convergence summary over generations: mean HV, mean IGD,
    mean GD, and mean archive/front size (all averaged across the traced
    runs), sharing one figure for a compact per-problem overview."""
    problem = result['problem']
    h = result['history']
    path = Path(fig_dir) / f"{problem['name']}_convergence.png"
    if h is None:
        return None
    it = h['iteration']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [('Mean hypervolume (higher = better)', h['hv'], '#1e8449'),
              ('Mean IGD (lower = better)', h['igd'], '#b03a2e'),
              ('Mean GD (lower = better)', h['gd'], '#2874a6'),
              ('Mean archive / front size', h['archive_size'], '#7d3c98')]
    for ax, (title, y, c) in zip(axes.ravel(), panels):
        ax.plot(it, y, color=c, lw=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('iteration')
        ax.grid(alpha=0.25)
    axes[1, 1].plot(it, h['front_size'], color='#f39c12', lw=1.2,
                    label='front size')
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"{problem['name']} — convergence "
                 f"(mean of {h['n_traced_runs']} runs)", fontsize=11)
    return _save(fig, path)


def plot_distribution(result, fig_dir):
    """Boxplot (+ jittered scatter overlay) of each of the 6 indicators'
    distribution across all independent runs -- shows run-to-run
    variability for this problem at a glance."""
    problem = result['problem']
    ms = result['run_metrics']
    path = Path(fig_dir) / f"{problem['name']}_distribution.png"
    keys = [('hv', 'Hypervolume'), ('igd', 'IGD'), ('gd', 'GD'),
            ('spacing', 'Spacing'), ('spread', 'Spread'), ('epsilon', 'Epsilon')]
    fig, axes = plt.subplots(1, 6, figsize=(15, 3.6))
    for ax, (k, title) in zip(axes, keys):
        vals = np.array([m[k] for m in ms], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.boxplot(vals, widths=0.55)
            ax.scatter(np.random.normal(1, 0.04, vals.size), vals,
                       s=8, alpha=0.5, color='#2471a3')
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.grid(alpha=0.25)
    fig.suptitle(f"{problem['name']} — indicator spread over "
                 f"{len(ms)} independent runs", fontsize=11)
    return _save(fig, path)


def plot_overview(all_results, fig_dir):
    """Single cross-problem horizontal bar chart: mean hypervolume (+/- std)
    for every problem in the sweep, color-coded by problem family
    (EvoPINN / Mixed / ZDT / DTLZ) -- the "at a glance" summary figure
    across the whole 27-problem benchmark."""
    
    path = Path(fig_dir) / "00_overview_hypervolume.png"
    names = list(all_results.keys())
    means = [all_results[n]['aggregate']['hv_mean'] for n in names]
    stds = [all_results[n]['aggregate']['hv_std'] for n in names]

    colors = [
        '#7d3c98' if n.startswith('EvoPINN') else
        '#c0392b' if n.startswith('Mixed') else
        '#1e8449' if n.startswith('ZDT') else
        '#2471a3'
        for n in names
    ]
    
    fig, ax = plt.subplots(figsize=(13, 6))
    y = np.arange(len(names))

    ax.barh(y, means, xerr=stds, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel('mean normalised hypervolume (higher = better)')
    ax.set_title('MOBCO — hypervolume across all problems')
    ax.grid(axis='x', alpha=0.25)

    return _save(fig, path)


def make_all_figures(result, fig_dir):
    """Generate every per-problem figure and return the list of successfully
    written file paths (skipping any that returned None, e.g. when no
    history was traced for this run)."""
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    made = [plot_front(result, fig_dir),               # expected vs. obtained
            plot_hypervolume(result, fig_dir),          # hypervolume vs. generation
            plot_objective_space(result, fig_dir),      # objective space (all runs)
            plot_decision_space(result, fig_dir),       # decision space (all runs)
            plot_convergence(result, fig_dir),          # convergence graph (HV/IGD/GD/size)
            plot_distribution(result, fig_dir)]         # boxplots over all runs
    return [m for m in made if m]
