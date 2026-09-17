from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class MOBCOConfig:
    """
    All tunable hyperparameters for one MOBCO experiment, grouped by the part
    of the algorithm they control. Two ready-made profiles are provided at
    the bottom of this file: QUICK (small/fast, for smoke-testing) and
    DEFAULT (the full experiment configuration).
    """
    #  MOBCO core 
    n_population: int = 300          # dogs + sheep per generation
    n_dogs: int = 3                  # number of leader agents ("dogs")
    archive_size: int = 300          # max size of the external non-dominated
                                      # archive (crowding-truncated)
    max_iterations: int = 1000        # >= 100 as required (project spec: >=200)
    n_runs: int = 50                 # >= 40 independent runs (project spec: >=50)
    base_seed: int = 1000            # run r uses seed = base_seed + r

    #  variation operator 
    mutation_rate: float = 0.1       # per-gene probability of Gaussian jitter
    eyeing_steps: int = 5            # stagnation length that triggers eyeing

    #  velocity / acceleration / time shaping (herding dynamics) 
    acc_damping: float = 0.9         # per-iteration momentum retained (acc *= this)
    mutation_decay_floor: float = 0.1   # mutation strength floor as run progresses (1.0 -> this)
    step_decay_floor: float = 0.1       # step/velocity-scale floor as run progresses
    stalking_pull: float = 2.0       # c1: pull strength for sheep far from their dog
    gathering_pull: float = 0.5      # c2: pull strength for sheep already near their dog

    #  metrics 
    hv_mc_samples_final: int = 60000   # Monte-Carlo samples, final HV (M >= 3)
    hv_mc_samples_history: int = 4000  # cheaper HV for the per-iteration trace
    ref_front_size: int = 1000         # points sampled from the true Pareto front
    hv_margin: float = 0.10            # reference point offset beyond the nadir

    #  bookkeeping 
    history_every: int = 1           # record convergence trace every g generations
    history_metric_runs: int = 10    # how many runs feed the averaged convergence
                                     # trace (HV per generation is the expensive
                                     # part; 10 runs is ample for a smooth curve)
    results_dir: str = "results"
    figures_dir: str = "figures"
    verbose: bool = True


    problems: list = field(default_factory=list)

    def ensure_dirs(self, root="."):
        """Create (if missing) and return the (results_dir, figures_dir) paths
        under `root`, so callers can write output files without checking
        existence themselves."""
        root = Path(root)
        for d in (self.results_dir, self.figures_dir):
            (root / d).mkdir(parents=True, exist_ok=True)
        return root / self.results_dir, root / self.figures_dir

    def as_dict(self):
        """Convert this config to a plain dict (e.g. for logging or saving
        alongside results)."""
        return asdict(self)


# A small profile handy for smoke-testing the pipeline before the real run.
QUICK = MOBCOConfig(n_population=40, n_dogs=3, archive_size=40,
                    max_iterations=10, n_runs=3, hv_mc_samples_final=5000,
                    hv_mc_samples_history=1000, ref_front_size=200)

DEFAULT = MOBCOConfig()
