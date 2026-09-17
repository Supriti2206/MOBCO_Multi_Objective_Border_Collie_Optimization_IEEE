"""
optimizer.py -- MOBCO (Multi-Objective Border Collie Optimization)

This is the main algorithm driver: for a given problem and config, it
initializes a population, then repeats the herding -> eyeing ->
velocity-update -> position-update -> mutation -> bounds-check ->
re-evaluate -> archive-update loop for `max_iterations` generations,
recording periodic history snapshots for convergence plots.
"""

import time
import numpy as np

from archive import Archive
from bounds_check import mobco_check
from dataset import bounds_arrays
from herding import mobco_herding
from population_init import mobco_generate
from velocity_time_acceleration import mobco_update_velocity
from position_update import mobco_update_position


class MOBCO:
    """One self-contained MOBCO run against a single problem/config/seed.
    Call `.run()` to execute the full optimization and get back a results
    dict (see `run()`'s return value)."""

    def __init__(self, problem, config, seed=0):
        self.problem = problem
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.lower, self.upper = bounds_arrays(problem)
        self.dim = problem['dim']
        self.n_obj = problem['n_obj']
        self.types = problem['objective_types']
        self.n_eval = 0   # running count of objective function evaluations (for reporting)

    def _evaluate(self, X):
        """Evaluate population - THIS IS YOUR FITNESS FUNCTION"""
        f = self.problem['evaluate']
        F = np.empty((len(X), self.n_obj))
        for i, x in enumerate(X):
            F[i] = f(x)
        self.n_eval += len(X)
        return F

    def _best_per_objective(self, F):
        """Direction-aware best value of every objective"""
        out = np.empty(self.n_obj)
        for j, t in enumerate(self.types):
            out[j] = F[:, j].min() if t == 'min' else F[:, j].max()
        return out

    def _mutate(self, population, progress=0.0):
        """Gaussian jitter mutation with exploration -> exploitation decay.

        Each decision variable of each individual is independently mutated
        with probability `rate` (which itself decays from
        cfg.mutation_rate down towards cfg.mutation_rate * floor as the run
        progresses), adding Gaussian noise scaled by the variable's range."""
        cfg = self.config
        floor = getattr(cfg, 'mutation_decay_floor', 0.1)
        decay = 1.0 - (1.0 - floor) * progress

        rate = cfg.mutation_rate * decay
        mask = self.rng.random(population.shape) < rate
        if not np.any(mask):
            return population
        span = self.upper - self.lower
        noise = self.rng.standard_normal(population.shape) * (0.1 * decay) * span[None, :]
        population = population.copy()
        population[mask] += noise[mask]
        return population

    def run(self):
        """
        MAIN MOBCO LOOP - This is the method that was missing!
        """
        cfg = self.config
        t0 = time.perf_counter()

        # Initialization
        population, Vt, acc, t = mobco_generate(
            cfg.n_population, self.dim, self.upper, self.lower, rng=self.rng)

        # Evaluate initial population
        fitness = self._evaluate(population)
        self.n_eval += cfg.n_population   # NOTE: _evaluate() already increments
                                           # self.n_eval by len(population) --
                                           # this line double-counts the
                                           # initial population's evaluations
                                           # in the reported n_eval total

        archive = Archive(self.types, max_size=cfg.archive_size)
        archive.update(population, fitness)

        hist = {'iteration': [], 'archive_size': [], 'front_size': [],
                'best': [], 'mean': [], 'snapshots': []}

        eye_counter = 0
        last_len = len(archive)

        for it in range(1, cfg.max_iterations + 1):
            # --- Step 1: Herding (sort by rank + crowding) ---
            # Reorders population/Vt/fitness/acc/t so the best individuals
            # come first -- this ordering is what defines who counts as a
            # "dog" (leader) vs "sheep" (follower) in the steps below.
            population, Vt, fitness, acc, t = mobco_herding(
                population, Vt, fitness, cfg.n_population, self.dim,
                acc, t, self.types)

            # --- Step 2: Eyeing (stagnation detection) ---
            # If the archive hasn't grown for `eyeing_steps` consecutive
            # iterations, set eye_flag=1 for this iteration -- this triggers
            # the direction-reversing behavior in velocity/position updates
            # meant to break the population out of a stall.
            eye_flag = 0
            cur_len = len(archive)
            if cur_len <= last_len:
                eye_counter += 1
                if eye_counter >= getattr(cfg, 'eyeing_steps', 5):
                    eye_flag = 1
                    eye_counter = 0
            else:
                eye_counter = 0
            last_len = cur_len

            progress = it / cfg.max_iterations   # 0 -> 1 over the run, used to decay step sizes

            # --- Step 3: Velocity update ---
            # Dogs (leaders) accelerate towards a sibling dog or a sparse
            # archive member; sheep (followers) are pulled towards their
            # nearest dog (see velocity_time_acceleration.py for the full
            # herding-behavior breakdown).
            Vt, acc, t, gathered_idx, stalked_idx = mobco_update_velocity(
                Vt, cfg.n_population, self.dim, acc, t,
                population, fitness, eye_flag, self.types,
                n_dogs=cfg.n_dogs, seed=self.rng,
                lower=self.lower, upper=self.upper, archive=archive,
                progress=progress, decay_floor=getattr(cfg, 'step_decay_floor', 0.1),
                acc_damping=getattr(cfg, 'acc_damping', 0.9),
                stalking_pull=getattr(cfg, 'stalking_pull', 2.0),
                gathering_pull=getattr(cfg, 'gathering_pull', 0.5))

            # --- Step 4: Position update ---
            # CAUTION: mobco_update_position computes the new position purely
            # from Vt/t/acc (v*t + 0.5*a*t^2) and does NOT add the current
            # `population` passed in here -- the return value directly
            # REPLACES `population` below. Worth double-checking this is the
            # intended kinematics for this "new" version (vs. the previously
            # identified bug of assigning displacement as an absolute
            # position rather than adding it to the current position -- see
            # position_update.py's docstring for more detail).
            population = mobco_update_position(
                population, Vt, t, acc, cfg.n_population, self.dim, eye_flag,
                n_dogs=cfg.n_dogs)

            # --- Step 5: Mutation ---
            population = self._mutate(population, progress=progress)

            # --- Step 6: Bounds check ---
            # Re-randomizes any position/acceleration/velocity cell (or, for
            # a non-finite time-step, the whole agent) that fell outside
            # [lower, upper] or went non-finite.
            population, acc, t, Vt = mobco_check(
                population, cfg.n_population, self.dim,
                self.upper, self.lower, acc, Vt, t, rng=self.rng)

            # --- Step 7: Re-evaluate fitness after position/mutation ---
            fitness = self._evaluate(population)

            # --- Step 8: Update archive ---
            # Merges this generation's population into the external archive
            # and re-filters/truncates it to stay non-dominated and within
            # max_size (see archive.py).
            archive.update(population, fitness)

            # --- Step 9: Record history ---
            # Only sampled every `history_every` iterations (plus the final
            # one) to keep memory/runtime bounded across many long runs.
            if (it % cfg.history_every == 0) or it == cfg.max_iterations:
                Ff = archive.fitness
                hist['iteration'].append(it)
                hist['archive_size'].append(len(archive))
                hist['front_size'].append(len(Ff))
                if len(Ff):
                    hist['best'].append(self._best_per_objective(Ff))
                    hist['mean'].append(Ff.mean(axis=0))
                else:
                    hist['best'].append(np.full(self.n_obj, np.nan))
                    hist['mean'].append(np.full(self.n_obj, np.nan))
                hist['snapshots'].append(Ff.copy())

        runtime = time.perf_counter() - t0

        return {
            'seed': self.seed,
            'archive_decisions': archive.decisions,
            'archive_fitness': archive.fitness,
            'front': archive.fitness.copy(),        # final non-dominated front (objective space)
            'front_decisions': archive.decisions.copy(),  # corresponding decision vectors
            'runtime': runtime,
            'n_eval': self.n_eval,
            'history': hist,
        }


# ----------------------------------------------------------------------
def run_single(problem, config, seed):
    """Convenience wrapper used by run_experiment.py: build a fresh MOBCO
    instance for this (problem, config, seed) and run it to completion."""
    return MOBCO(problem, config, seed=seed).run()