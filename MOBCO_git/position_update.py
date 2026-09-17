import numpy as np

def mobco_update_position(pop, Vt, t, acc, n, L, eye_flag, n_dogs=3):
    """
    Kinematic position update step of MOBCO: for each individual, compute a
    new position from the standard 1D kinematics equation
        x = v*t + 0.5*a*t^2
    applied independently per decision variable.

    Parameters:
        pop: (n, L) current population positions -- NOTE: not referenced in
             the formula below (see caution note under Returns)
        Vt: (n, L) current velocities
        t: (n,) or (n, L) elapsed "time" step per individual
        acc: (n, L) current accelerations
        n: population size
        L: number of decision variables (dimensionality)
        eye_flag: 1 if the "eyeing"/stagnation behavior is active this
                  iteration (only affects the non-dog individuals below),
                  0 otherwise
        n_dogs: number of lead "dogs" (the first n_dogs individuals in the
                herded/sorted population, i.e. the best-ranked ones) that
                always use the standard `+` kinematics regardless of eye_flag

    Behavior:
        - The first n_dogs individuals (best/lead individuals after herding)
          always move using x = v*t + 0.5*a*t^2 (added acceleration term).
        - The remaining individuals ("pack") use the same formula when
          eye_flag == 0, but flip the acceleration term's sign
          (x = v*t - 0.5*a*t^2) when eye_flag == 1, i.e. while the archive
          is stagnating -- this decelerates/reverses their exploratory push
          to encourage a change in search direction.

    Returns:
        pop_new: (n, L) new position for every individual.

    CAUTION (worth double-checking against the algorithm description/paper):
        pop_new is computed purely from Vt, t, acc -- the incoming `pop`
        array is never added in. In optimizer.py this return value directly
        replaces the population (`population = mobco_update_position(...)`),
        so the new position is treated as an ABSOLUTE position, not a
        displacement added to the previous one. This is the same failure
        pattern as a previously-identified bug in this codebase (displacement
        assigned as absolute position instead of being added to the current
        position) -- verify whether this is intentional for this "new"
        version (e.g. Vt/t/acc are meant to already encode absolute state)
        or whether `pop[i] +` is still missing here.
    """
    pop_new = np.zeros((n, L))

    for i in range(n):
        if i < n_dogs:  # First n_dogs individuals = lead "dogs" (best-ranked after herding)
            pop_new[i] = Vt[i] * t[i] + 0.5 * acc[i] * (t[i] ** 2)
        else:
            if eye_flag == 1:
                # Stagnation detected: reverse the acceleration contribution
                # for pack members to break out of the stall
                pop_new[i] = Vt[i] * t[i] - 0.5 * acc[i] * (t[i] ** 2)
            else:
                pop_new[i] = Vt[i] * t[i] + 0.5 * acc[i] * (t[i] ** 2)

    return pop_new