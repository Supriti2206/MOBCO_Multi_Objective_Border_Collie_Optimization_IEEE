import numpy as np


def mobco_update_velocity(Vt, n, L, acc, t, population, fitness,
                           eye_flag, objective_types, n_dogs=3, seed=None,
                           lower=None, upper=None, archive=None,
                           progress=0.0, decay_floor=0.1, acc_damping=0.9,
                           stalking_pull=2.0, gathering_pull=0.5):
    """
    Update velocity, acceleration and time-step arrays for one generation.

    This implements the "herding" behavior of Border Collie Optimization in
    velocity/acceleration space: the population is assumed already sorted
    (see herding.py) so the first `n_dogs` individuals are the current best
    ("dogs"/leaders) and the rest are "sheep" (followers).

    Parameters:
        Vt: (n, L) current velocities
        n: population size
        L: number of decision variables
        acc: (n, L) current accelerations
        t: (n,) current per-agent time-step (read for shape only; a fresh
           t_new is generated inside this function)
        population: (n, L) current positions, assumed pre-sorted best-first
        fitness: (n, n_obj) current fitness values (not directly used here,
                 but accepted for a consistent call signature / potential
                 future use)
        eye_flag: 1 if stagnation ("eyeing") behavior is active this
                  iteration (not used directly in this function -- consumed
                  downstream in position_update.py)
        objective_types: list like ['min', 'max', ...] (accepted for a
                          consistent call signature; not used directly here)
        n_dogs: number of leaders at the front of `population`
        seed: an int seed, an existing np.random.Generator, or None
        lower, upper: search-space bounds, used to scale step/velocity limits
                      relative to each dimension's range
        archive: the current external Archive (used by dogs for the
                 diversity-seeking behavior below)
        progress: fraction of the run completed (0 at start, 1 at the final
                  iteration), used to decay exploration over time
        decay_floor: minimum fraction of the original step/velocity scale
                     retained even at progress == 1 (prevents step size from
                     decaying all the way to zero)
        acc_damping: multiplicative damping applied to the incoming
                     acceleration each generation (simulates friction/decay)
        stalking_pull: pull-strength coefficient for sheep that are farther
                       than the median distance from their nearest dog
                       ("stalking" -- cautious approach)
        gathering_pull: pull-strength coefficient for sheep already close to
                        their nearest dog ("gathering" -- being herded in)

    Returns:
        (Vt_new, acc_new, t_new, gathered_idx, stalked_idx):
            Vt_new, acc_new: updated (n, L) velocity/acceleration arrays
            t_new: freshly sampled (n,) time-step for this generation
            gathered_idx, stalked_idx: index lists (into the sheep portion of
                the population) recording which behavior each sheep took,
                useful for diagnostics/visualization
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    # decay goes from 1.0 (progress=0, start of run) down to decay_floor
    # (progress=1, end of run) -- shrinks step/velocity scales over time
    decay = 1.0 - (1.0 - decay_floor) * progress

    # ---- Compute step scale from bounds, decayed by generation progress ----
    if lower is not None and upper is not None:
        span = np.asarray(upper) - np.asarray(lower)
        step_scale = 0.05 * decay * span   # shrinks from 5% -> 0.5% of range
        v_max = 0.2 * decay * span         # shrinks from 20% -> 2% velocity
    else:
        # No bounds given: fall back to fixed (non-range-relative) scales
        step_scale = np.full(L, 0.05 * decay)
        v_max = np.full(L, 0.2 * decay)

    acc = acc_damping * acc  # simulate friction: shrink last generation's acceleration

    dogs = population[:n_dogs]     # leader individuals (best-ranked, from herding.py's sort)
    n_sheep = n - n_dogs            # remaining "follower" individuals

    acc_new = acc.copy()
    t_new = rng.random(n)
    t_new[t_new == 0] = 1e-6  # avoid a zero time-step (would freeze that agent)

    # ---- Dogs: accelerate towards sibling leader OR sparse archive member ----
    if n_dogs > 1:
        for i in range(n_dogs):
            # 30% chance: target a sparse (isolated) archive member instead of
            # a sibling dog -- this is the diversity-injection mechanism that
            # pulls leaders toward under-explored regions of the Pareto front
            if archive is not None and len(archive) > n_dogs and rng.random() < 0.3:
                arch_fit = archive.fitness
                arch_dec = archive.decisions

                if len(arch_fit) > 1:
                    n_arch = len(arch_fit)
                    # Compute all pairwise distances between archive members
                    # in OBJECTIVE space (not decision space) to find isolated points
                    dist_matrix = np.full((n_arch, n_arch), np.inf)
                    for a in range(n_arch):
                        diff = arch_fit - arch_fit[a]
                        dist_matrix[a] = np.sqrt((diff ** 2).sum(axis=1))
                        dist_matrix[a, a] = np.inf  # exclude self-distance (always 0)
                    min_dists = dist_matrix.min(axis=1)  # distance to each point's nearest neighbor
                    # Pick from the top 50% most isolated (sparsest) archive members
                    threshold = np.median(min_dists)
                    candidates = np.where(min_dists >= threshold)[0]
                    if len(candidates) > 0:
                        chosen = rng.choice(candidates)
                        target = arch_dec[chosen]  # target is a DECISION vector, for position steering
                    else:
                        target = arch_dec[0]
                else:
                    target = arch_dec[0]
            else:
                # Standard: target a random sibling dog (peer-following behavior)
                others = [k for k in range(n_dogs) if k != i]
                target = dogs[rng.choice(others)]

            # Steer acceleration toward the chosen target: unit direction
            # vector scaled by a random magnitude and the (decayed) step scale
            direction = target - population[i]
            dist = np.linalg.norm(direction)
            if dist > 1e-12:
                direction = direction / dist
                magnitude = rng.random() * 0.5 + 0.5  # Random in [0.5, 1.0]
                acc_new[i] = acc[i] + magnitude * direction * step_scale
            else:
                # If already at target, add small random exploration instead
                # of a zero-direction (no-op) update
                acc_new[i] = acc[i] + rng.standard_normal(L) * step_scale * 0.1

    elif n_dogs == 1:
        # With only one leader there's no sibling/archive-target logic --
        # just nudge it with scaled random acceleration
        acc_new[0] = acc[0] + rng.random(L) * step_scale * rng.normal(0, 0.1, L)

    # ---- Sheep: herded towards nearest dog, stalking vs gathering ----
    gathered_idx, stalked_idx = [], []
    if n_sheep > 0:
        sheep_pos = population[n_dogs:]
        # Pairwise distances from every sheep to every dog (vectorized)
        diffs = sheep_pos[:, None, :] - dogs[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        nearest_dog = np.argmin(dists, axis=1)                       # index of closest dog, per sheep
        nearest_dist = dists[np.arange(n_sheep), nearest_dog]        # that distance

        # Sheep farther than the median distance are "stalked" (approached
        # cautiously with a stronger pull); closer ones are "gathered" (herded
        # in gently) -- this median split is recomputed fresh every generation
        threshold = np.median(nearest_dist) if n_sheep > 1 else nearest_dist[0]

        c1 = stalking_pull    # stalking pull strength (far sheep)
        c2 = gathering_pull   # gathering pull strength (near sheep)

        for j in range(n_sheep):
            gi = n_dogs + j  # this sheep's index in the full population array
            target = dogs[nearest_dog[j]]
            direction = target - population[gi]
            dist = np.linalg.norm(direction)
            if dist > 1e-12:
                direction = direction / dist

            if nearest_dist[j] > threshold:
                stalked_idx.append(gi)
                acc_new[gi] = acc[gi] + c1 * rng.random(L) * direction * step_scale
            else:
                gathered_idx.append(gi)
                acc_new[gi] = acc[gi] + c2 * rng.random(L) * direction * step_scale

    # ---- Velocity update WITH CLAMPING ----
    # v_new = v_old + a*t, then clamp to +/- v_max per dimension so velocity
    # can't blow up and cause an unbounded position jump next step
    Vt_new = Vt + acc_new * t_new[:, None]
    Vt_new = np.clip(Vt_new, -v_max[None, :], v_max[None, :])

    return Vt_new, acc_new, t_new, gathered_idx, stalked_idx