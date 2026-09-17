"""
evopinn_problem.py -- 12 "EvoPINN" benchmark problems: multi-objective
formulations of Physics-Informed Neural Network (PINN) training.

Each problem trains a small tanh-MLP to approximate the solution u(x) or
u(x,t) of a known PDE/ODE. Instead of collapsing the usual PINN loss terms
(PDE residual, boundary condition, initial condition, data-fit) into one
scalar via fixed weights, each term is kept as its OWN objective -- so a
multi-objective optimizer must trade off residual accuracy vs. boundary
accuracy vs. initial-condition accuracy vs. data-fit, rather than the
weights being hand-tuned in advance. Every problem has exactly 4 objectives
(n_obj = 4): [L_residual, L_boundary, L_initial, L_data], all 'min'.

The decision vector `theta` is the MLP's flattened weights+biases (so
`dim` = the network's parameter count) -- this is why these problems have
much higher dimensionality than the ZDT/DTLZ problems.

No ML libraries are used: TanhMLP below hand-implements both a forward pass
and forward-mode automatic differentiation (up to 2nd derivatives) so that
PDE residuals involving u_x, u_xx, u_t, etc. can be computed exactly without
autograd/PyTorch/TensorFlow.
"""

import numpy as np


# Shared PINN machinery (hand-coded forward-mode autodiff, no ML libs)
class TanhMLP:
    """
    A simple fully-connected feed-forward network with tanh activations on
    every hidden layer and a linear (no activation) output layer.

    layer_sizes: e.g. [2, 10, 10, 1] = 2 inputs -> 10 -> 10 -> 1 output.
    All parameters (every layer's W and b) are packed into a single flat
    vector `theta`, which is what the multi-objective optimizer actually
    searches over -- `unpack` reconstructs the individual weight matrices
    and bias vectors from that flat vector.
    """

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.n_params = sum(
            layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )

    def unpack(self, theta):
        """Slice the flat parameter vector `theta` into a list of (W, b)
        pairs, one per layer, in order."""
        Ws, bs, idx = [], [], 0
        for i in range(len(self.layer_sizes) - 1):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            w_size = n_in * n_out
            W = theta[idx: idx + w_size].reshape(n_out, n_in); idx += w_size
            b = theta[idx: idx + n_out]; idx += n_out
            Ws.append(W); bs.append(b)
        return Ws, bs

    def forward(self, theta, X):
        """Standard forward pass: X (N, n_in) -> output (N, n_out).
        tanh activation on every layer except the last (linear output)."""
        Ws, bs = self.unpack(theta)
        a = X
        for l, (W, b) in enumerate(zip(Ws, bs)):
            z = a @ W.T + b
            a = np.tanh(z) if l < len(Ws) - 1 else z
        return a

    def forward_with_derivs(self, theta, X, deriv_dims):
        """
        Forward pass that ALSO computes exact 1st (and, where requested,
        2nd) partial derivatives of the network output with respect to
        selected input dimensions, using forward-mode differentiation
        propagated layer by layer alongside the normal activations.

        Parameters:
            theta: flat parameter vector
            X: (N, n_in) inputs
            deriv_dims: dict mapping input-dimension index -> derivative
                        order needed for that dimension (1 or 2), e.g.
                        {0: 2, 1: 1} means "give me d/dx0 and d^2/dx0^2, and
                        d/dx1" (used e.g. for a PDE needing u_xx and u_t)

        Returns:
            (a, derivs): `a` is the normal forward-pass output (N, n_out);
            `derivs` is a dict keyed by (dim, order) -> (N, n_out) array of
            that partial derivative, e.g. derivs[(0, 2)] = d^2 u / dx0^2.

        How it works: cur_first[d] tracks d(activation)/d(x_d) as it's
        propagated through each layer using the chain rule (for a linear
        layer, just apply W; for tanh, multiply by sech^2). cur_second[d]
        similarly tracks the second derivative, using the tanh second-
        derivative identity (tanh)'' = -2*tanh*sech^2 combined with the
        product rule for the composition.
        """
        Ws, bs = self.unpack(theta)
        N, n_in = X.shape
        # Seed: d(x_d)/d(x_d) = 1, everything else 0 (standard forward-mode
        # "tangent" initialization, one input dimension at a time)
        cur_first = {d: np.zeros((N, n_in)) for d in deriv_dims}
        for d in deriv_dims:
            cur_first[d][:, d] = 1.0
        cur_second = {d: np.zeros((N, n_in)) for d in deriv_dims if deriv_dims[d] >= 2}
        a = X
        for l, (W, b) in enumerate(zip(Ws, bs)):
            z = a @ W.T + b
            z_first = {d: cur_first[d] @ W.T for d in deriv_dims}     # d(z)/d(x_d) = d(a)/d(x_d) @ W.T (linear layer, chain rule)
            z_second = {d: cur_second[d] @ W.T for d in cur_second}   # same for the second derivative (W is linear, so no extra term)
            if l == len(Ws) - 1:
                # Output layer is linear (no activation) -- derivatives pass through unchanged
                a = z
                cur_first, cur_second = z_first, z_second
            else:
                t = np.tanh(z)
                sech2 = 1 - t**2                 # d(tanh)/dz = sech^2(z) = 1 - tanh^2(z)
                tanh_dbl = -2 * t * sech2         # d^2(tanh)/dz^2 = -2*tanh(z)*sech^2(z)
                new_first = {d: sech2 * z_first[d] for d in deriv_dims}
                # Second derivative via chain rule on a composition:
                # d^2(tanh(z))/dx^2 = tanh''(z)*(dz/dx)^2 + tanh'(z)*(d^2z/dx^2)
                new_second = {d: tanh_dbl * z_first[d]**2 + sech2 * z_second[d]
                              for d in cur_second}
                a = t
                cur_first, cur_second = new_first, new_second
        derivs = {(d, 1): cur_first[d] for d in deriv_dims}
        derivs.update({(d, 2): cur_second[d] for d in cur_second})
        return a, derivs


def mse(a, b=0.0):
    """Mean squared error between `a` and `b` (or `a` and 0 if b omitted).
    Used as the loss form for every PINN objective term below."""
    return float(np.mean((a - b) ** 2))


def rk4_integrate(f, y0, t0, t1, n_steps):
    """Classic 4th-order Runge-Kutta integrator: integrate dy/dt = f(t, y)
    from t0 to t1 in n_steps fixed steps, starting at y0. Used to generate
    ground-truth trajectories (for the data-fit term) for ODE problems
    without a closed-form solution (Van der Pol, Lotka-Volterra)."""
    ts = np.linspace(t0, t1, n_steps + 1)
    h = ts[1] - ts[0]
    ys = np.zeros((n_steps + 1, len(y0)))
    y = np.array(y0, dtype=float)
    ys[0] = y
    for i in range(n_steps):
        t = ts[i]
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[i + 1] = y
    return ts, ys


def unif(rng, lo, hi, n, d=1):
    """Shorthand for n uniform-random samples in [lo, hi], shaped (n, d) --
    used throughout to build each problem's collocation/boundary/data
    point sets."""
    return rng.uniform(lo, hi, size=(n, d))


# EvoPINNProblem
class EvoPINNProblem:
    """
    Base class for all 12 EvoPINN problems. Each subclass must implement:
      - _build_data(): sample the residual/boundary/initial/data point sets
                        for this specific PDE/ODE (self.Xr, self.Xb, ... )
      - _objectives(theta): compute [L_residual, L_boundary, L_initial,
                             L_data] for a given parameter vector

    Common setup handled here: builds the TanhMLP for `net_layers`, derives
    `dim` from the network's actual parameter count (not hand-specified),
    and fixes n_obj=4 / all-'min' objective_types for every subclass.
    """

    def __init__(self, name, net_layers, seed=0, param_bounds=(-2.0, 2.0)):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.net = TanhMLP(net_layers)
        self.n_obj = 4  # residual, bc, ic, data — fixed across all 12
        self.dim = self.net.n_params  # DERIVED, not assigned
        self.bounds = [param_bounds] * self.dim
        self.objective_types = ['min'] * self.n_obj
        self._build_data()

    def _build_data(self):
        raise NotImplementedError

    def evaluate(self, x):
        return self._objectives(np.asarray(x))

    def _objectives(self, theta):
        raise NotImplementedError


class EvoPINN1_Poisson1D(EvoPINNProblem):
    """1D Poisson equation: u_xx = f(x), on x in [0,1], with a manufactured
    solution u(x) = sin(pi*x) + 0.5*x^2 (steady-state, no time dependence
    -> L_ic is fixed at 0)."""

    def __init__(self, seed=0):
        super().__init__("EvoPINN1_Poisson1D", [1, 10, 10, 1], seed)

    def _build_data(self):
        r = self.rng
        self.Xr = unif(r, 0, 1, 100)
        self.Xb = np.array([[0.0], [1.0]])
        self.ub = np.sin(np.pi * self.Xb) + 0.5 * self.Xb**2
        self.Xd = unif(r, 0, 1, 10)
        self.ud = np.sin(np.pi * self.Xd) + 0.5 * self.Xd**2

    @staticmethod
    def f(x):
        return -np.pi**2 * np.sin(np.pi * x) + 1.0

    def _objectives(self, theta):
        _, d = self.net.forward_with_derivs(theta, self.Xr, {0: 2})
        r = d[(0, 2)] - self.f(self.Xr)
        L_r = mse(r)
        L_bc = mse(self.net.forward(theta, self.Xb), self.ub)
        L_ic = 0.0
        L_d = mse(self.net.forward(theta, self.Xd), self.ud)
        return np.array([L_r, L_bc, L_ic, L_d])


class EvoPINN2_Heat1D(EvoPINNProblem):
    """1D heat/diffusion equation: u_t = alpha * u_xx, x in [0,1], t in
    [0,1], with initial condition u(x,0) = sin(pi*x) and homogeneous
    (u=0) boundary conditions."""
    alpha = 0.1

    def __init__(self, seed=1):
        super().__init__("EvoPINN2_Heat1D", [2, 10, 10, 1], seed)

    def _build_data(self):
        r = self.rng
        self.Xr = np.hstack([unif(r, 0, 1, 100), unif(r, 0, 1, 100)])
        tb = unif(r, 0, 1, 20)
        self.Xb = np.vstack([np.hstack([np.zeros_like(tb), tb]),
                              np.hstack([np.ones_like(tb), tb])])
        self.ub = np.zeros((self.Xb.shape[0], 1))
        x0 = unif(r, 0, 1, 20)
        self.X0 = np.hstack([x0, np.zeros_like(x0)])
        self.u0 = np.sin(np.pi * x0)
        self.Xd = np.hstack([unif(r, 0, 1, 10), unif(r, 0, 1, 10)])
        self.ud = self._exact(self.Xd)

    def _exact(self, X):
        x, t = X[:, :1], X[:, 1:]
        return np.exp(-self.alpha * np.pi**2 * t) * np.sin(np.pi * x)

    def _objectives(self, theta):
        _, d = self.net.forward_with_derivs(theta, self.Xr, {0: 2, 1: 1})
        r = d[(1, 1)] - self.alpha * d[(0, 2)]
        L_r = mse(r)
        L_bc = mse(self.net.forward(theta, self.Xb), self.ub)
        L_ic = mse(self.net.forward(theta, self.X0), self.u0)
        L_d = mse(self.net.forward(theta, self.Xd), self.ud)
        return np.array([L_r, L_bc, L_ic, L_d])


class EvoPINN3_Advection1D(EvoPINNProblem):
    """1D linear advection equation: u_t + c*u_x = 0, x in [0, 2*pi],
    periodic-style boundary matching, with initial condition
    u(x,0) = sin(x); exact solution is the traveling wave sin(x - c*t)."""
    c = 1.0

    def __init__(self, seed=2):
        super().__init__("EvoPINN3_Advection1D", [2, 10, 10, 1], seed)

    def _build_data(self):
        r = self.rng
        self.Xr = np.hstack([unif(r, 0, 2 * np.pi, 100), unif(r, 0, 1, 100)])
        tb = unif(r, 0, 1, 20)
        self.Xb = np.vstack([np.hstack([np.zeros_like(tb), tb]),
                              np.hstack([np.full_like(tb, 2 * np.pi), tb])])
        self.ub = self._exact(self.Xb)
        x0 = unif(r, 0, 2 * np.pi, 20)
        self.X0 = np.hstack([x0, np.zeros_like(x0)])
        self.u0 = np.sin(x0)
        self.Xd = np.hstack([unif(r, 0, 2 * np.pi, 10), unif(r, 0, 1, 10)])
        self.ud = self._exact(self.Xd)

    def _exact(self, X):
        x, t = X[:, :1], X[:, 1:]
        return np.sin(x - self.c * t)

    def _objectives(self, theta):
        _, d = self.net.forward_with_derivs(theta, self.Xr, {0: 1, 1: 1})
        r = d[(1, 1)] + self.c * d[(0, 1)]
        L_r = mse(r)
        L_bc = mse(self.net.forward(theta, self.Xb), self.ub)
        L_ic = mse(self.net.forward(theta, self.X0), self.u0)
        L_d = mse(self.net.forward(theta, self.Xd), self.ud)
        return np.array([L_r, L_bc, L_ic, L_d])


class EvoPINN4_Wave1D(EvoPINNProblem):
    """1D wave equation: u_tt = c^2 * u_xx, x in [0,1], with initial
    displacement u(x,0)=sin(pi*x) and initial velocity u_t(x,0)=0 (so the
    initial-condition loss L_ic combines both a value term and a velocity
    term, weighted 0.5/0.5)."""
    c = 1.0

    def __init__(self, seed=3):
        super().__init__("EvoPINN4_Wave1D", [2, 10, 10, 1], seed)

    def _build_data(self):
        r = self.rng
        self.Xr = np.hstack([unif(r, 0, 1, 100), unif(r, 0, 1, 100)])
        tb = unif(r, 0, 1, 20)
        self.Xb = np.vstack([np.hstack([np.zeros_like(tb), tb]),
                              np.hstack([np.ones_like(tb), tb])])
        self.ub = np.zeros((self.Xb.shape[0], 1))
        self.x0 = unif(r, 0, 1, 20)
        self.X0 = np.hstack([self.x0, np.zeros_like(self.x0)])
        self.u0_val = np.sin(np.pi * self.x0)
        self.u0_vel = np.zeros_like(self.x0)
        self.Xd = np.hstack([unif(r, 0, 1, 10), unif(r, 0, 1, 10)])
        self.ud = self._exact(self.Xd)

    def _exact(self, X):
        x, t = X[:, :1], X[:, 1:]
        return np.sin(np.pi * x) * np.cos(np.pi * self.c * t)

    def _objectives(self, theta):
        _, d = self.net.forward_with_derivs(theta, self.Xr, {0: 2, 1: 2})
        r = d[(1, 2)] - self.c**2 * d[(0, 2)]
        L_r = mse(r)
        L_bc = mse(self.net.forward(theta, self.Xb), self.ub)
        _, d0 = self.net.forward_with_derivs(theta, self.X0, {1: 1})
        u0_pred = self.net.forward(theta, self.X0)
        L_ic = 0.5 * mse(u0_pred, self.u0_val) + 0.5 * mse(d0[(1, 1)], self.u0_vel)
        L_d = mse(self.net.forward(theta, self.Xd), self.ud)
        return np.array([L_r, L_bc, L_ic, L_d])


class EvoPINN5_Burgers1D(EvoPINNProblem):
    """Viscous (forced) Burgers' equation: u_t + u*u_x = nu*u_xx + f(x,t),
    x in [0,1]; f is derived so the manufactured solution
    u(x,t) = sin(pi*x)*exp(-t) satisfies the PDE exactly."""
    nu = 0.01 / np.pi

    def __init__(self, seed=4):
        super().__init__("EvoPINN5_Burgers1D", [2, 10, 10, 1], seed)

    def _build_data(self):
        r = self.rng
        self.Xr = np.hstack([unif(r, 0, 1, 100), unif(r, 0, 1, 100)])
        tb = unif(r, 0, 1, 20)
        self.Xb = np.vstack([np.hstack([np.zeros_like(tb), tb]),
                              np.hstack([np.ones_like(tb), tb])])
        self.ub = np.zeros((self.Xb.shape[0], 1))
        x0 = unif(r, 0, 1, 20)
        self.X0 = np.hstack([x0, np.zeros_like(x0)])
        self.u0 = np.sin(np.pi * x0)
        self.Xd = np.hstack([unif(r, 0, 1, 10), unif(r, 0, 1, 10)])
        self.ud = self._exact(self.Xd)

    def _exact(self, X):
        x, t = X[:, :1], X[:, 1:]
        return np.sin(np.pi * x) * np.exp(-t)

    def _f(self, X):
        x, t = X[:, :1], X[:, 1:]
        u = self._exact(X)
        return u * (self.nu * np.pi**2 - 1.0) + 0.5 * np.pi * np.sin(2 * np.pi * x) * np.exp(-2 * t)

    def _objectives(self, theta):
        u, d = self.net.forward_with_derivs(theta, self.Xr, {0: 2, 1: 1})
        r = d[(1, 1)] + u * d[(0, 1)] - self.nu * d[(0, 2)] - self._f(self.Xr)
        L_r = mse(r)
        L_bc = mse(self.net.forward(theta, self.Xb), self.ub)
        L_ic = mse(self.net.forward(theta, self.X0), self.u0)
        L_d = mse(self.net.forward(theta, self.Xd), self.ud)
        return np.array([L_r, L_bc, L_ic, L_d])


class EvoPINN6_AllenCahn1D(EvoPINNProblem):
    """Forced Allen-Cahn (reaction-diffusion phase-field) equation:
    u_t = d_coef*u_xx + u - u^3 + f(x,t), x in [-1,1]; f derived so the
    manufactured solution u(x,t) = sin(pi*x)*exp(-t) satisfies the PDE
    exactly."""
    d_coef = 0.1

    def __init__(self, seed=5):
        super().__init__("EvoPINN6_AllenCahn1D", [2, 10, 10, 1], seed)

    def _build_data(self):
        r = self.rng
        self.Xr = np.hstack([unif(r, -1, 1, 100), unif(r, 0, 1, 100)])
        tb = unif(r, 0, 1, 20)
        self.Xb = np.vstack([np.hstack([-np.ones_like(tb), tb]),
                              np.hstack([np.ones_like(tb), tb])])
        self.ub = np.zeros((self.Xb.shape[0], 1))
        x0 = unif(r, -1, 1, 20)
        self.X0 = np.hstack([x0, np.zeros_like(x0)])
        self.u0 = np.sin(np.pi * x0)
        self.Xd = np.hstack([unif(r, -1, 1, 10), unif(r, 0, 1, 10)])
        self.ud = self._exact(self.Xd)

    def _exact(self, X):
        x, t = X[:, :1], X[:, 1:]
        return np.sin(np.pi * x) * np.exp(-t)

    def _f(self, X):
        u = self._exact(X)
        return u * (self.d_coef * np.pi**2 - 2.0) + u**3

    def _objectives(self, theta):
        u, d = self.net.forward_with_derivs(theta, self.Xr, {0: 2, 1: 1})
        r = d[(1, 1)] - self.d_coef * d[(0, 2)] - u + u**3 - self._f(self.Xr)
        L_r = mse(r)
        L_bc = mse(self.net.forward(theta, self.Xb), self.ub)
        L_ic = mse(self.net.forward(theta, self.X0), self.u0)
        L_d = mse(self.net.forward(theta, self.Xd), self.ud)
        return np.array([L_r, L_bc, L_ic, L_d])


class EvoPINN7_Poisson2D(EvoPINNProblem):
    """2D Poisson equation: u_xx + u_yy = f(x,y), (x,y) in [0,1]^2, with
    manufactured solution u(x,y) = sin(pi*x)*sin(pi*y) and zero boundary
    on all 4 edges of the unit square (steady-state -> L_ic fixed at 0)."""

    def __init__(self, seed=6):
        super().__init__("EvoPINN7_Poisson2D", [2, 10, 10, 1], seed)

    def _build_data(self):
        r = self.rng
        self.Xr = np.hstack([unif(r, 0, 1, 150), unif(r, 0, 1, 150)])
        edge = unif(r, 0, 1, 20)
        z, o = np.zeros_like(edge), np.ones_like(edge)
        self.Xb = np.vstack([np.hstack([z, edge]), np.hstack([o, edge]),
                              np.hstack([edge, z]), np.hstack([edge, o])])
        self.ub = np.zeros((self.Xb.shape[0], 1))
        self.Xd = np.hstack([unif(r, 0, 1, 10), unif(r, 0, 1, 10)])
        self.ud = self._exact(self.Xd)

    def _exact(self, X):
        x, y = X[:, :1], X[:, 1:]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def _f(self, X):
        return -2 * np.pi**2 * self._exact(X)

    def _objectives(self, theta):
        _, d = self.net.forward_with_derivs(theta, self.Xr, {0: 2, 1: 2})
        r = d[(0, 2)] + d[(1, 2)] - self._f(self.Xr)
        L_r = mse(r)
        L_bc = mse(self.net.forward(theta, self.Xb), self.ub)
        L_ic = 0.0
        L_d = mse(self.net.forward(theta, self.Xd), self.ud)
        return np.array([L_r, L_bc, L_ic, L_d])


class EvoPINN8_ReactionDiffusion1D(EvoPINNProblem):
    """Forced reaction-diffusion equation: u_t = D*u_xx + k*u*(1-u) +
    f(x,t), x in [0,1]; f derived so the manufactured solution
    u(x,t) = sin(pi*x)*exp(-t) satisfies the PDE exactly."""
    D, k = 0.05, 1.0

    def __init__(self, seed=7):
        super().__init__("EvoPINN8_ReactionDiffusion1D", [2, 10, 10, 1], seed)

    def _build_data(self):
        r = self.rng
        self.Xr = np.hstack([unif(r, 0, 1, 100), unif(r, 0, 1, 100)])
        tb = unif(r, 0, 1, 20)
        self.Xb = np.vstack([np.hstack([np.zeros_like(tb), tb]),
                              np.hstack([np.ones_like(tb), tb])])
        self.ub = np.zeros((self.Xb.shape[0], 1))
        x0 = unif(r, 0, 1, 20)
        self.X0 = np.hstack([x0, np.zeros_like(x0)])
        self.u0 = np.sin(np.pi * x0)
        self.Xd = np.hstack([unif(r, 0, 1, 10), unif(r, 0, 1, 10)])
        self.ud = self._exact(self.Xd)

    def _exact(self, X):
        x, t = X[:, :1], X[:, 1:]
        return np.sin(np.pi * x) * np.exp(-t)

    def _f(self, X):
        u = self._exact(X)
        return u * (self.D * np.pi**2 - 1.0 - self.k) + self.k * u**2

    def _objectives(self, theta):
        u, d = self.net.forward_with_derivs(theta, self.Xr, {0: 2, 1: 1})
        r = d[(1, 1)] - self.D * d[(0, 2)] - self.k * u * (1 - u) - self._f(self.Xr)
        L_r = mse(r)
        L_bc = mse(self.net.forward(theta, self.Xb), self.ub)
        L_ic = mse(self.net.forward(theta, self.X0), self.u0)
        L_d = mse(self.net.forward(theta, self.Xd), self.ud)
        return np.array([L_r, L_bc, L_ic, L_d])


class EvoPINN9_DampedOscillator(EvoPINNProblem):
    """Damped harmonic oscillator ODE: u'' + 2*zeta*w*u' + w^2*u = 0, t in
    [0,4], with initial conditions u(0)=1, u'(0)=0. Has a closed-form exact
    solution (used directly for the data-fit term, no numerical
    integration needed)."""

    zeta = 0.2
    w = 2 * np.pi

    def __init__(self, seed=8):
        super().__init__(
            "EvoPINN9_DampedOscillator",
            [1, 10, 10, 1],
            seed
        )

    def _build_data(self):
        self.wd = self.w * np.sqrt(1 - self.zeta**2)

        r = self.rng
        self.Tr = unif(r, 0, 4, 100)
        self.T0 = np.array([[0.0]])
        self.Td = unif(r, 0, 4, 10)
        self.ud = self._exact(self.Td)

    def _exact(self, T):
        z, w, wd = self.zeta, self.w, self.wd

        return np.exp(-z * w * T) * (
            np.cos(wd * T)
            + (z * w / wd) * np.sin(wd * T)
        )

    def _objectives(self, theta):
        _, d = self.net.forward_with_derivs(
            theta,
            self.Tr,
            {0: 2}
        )

        r = (
            d[(0, 2)]
            + 2 * self.zeta * self.w * d[(0, 1)]
            + self.w**2 * self.net.forward(theta, self.Tr)
        )

        L_r = mse(r)

        u0 = self.net.forward(theta, self.T0)

        _, d0 = self.net.forward_with_derivs(
            theta,
            self.T0,
            {0: 1}
        )

        L_bc = mse(u0, 1.0)
        L_ic = mse(d0[(0, 1)], 0.0)
        L_d = mse(
            self.net.forward(theta, self.Td),
            self.ud
        )

        return np.array([L_r, L_bc, L_ic, L_d])

class EvoPINN10_VanDerPol(EvoPINNProblem):
    """Van der Pol oscillator ODE: u'' - mu*(1-u^2)*u' + u = 0, t in
    [0,10], initial conditions u(0)=2, u'(0)=0. No closed-form solution --
    ground-truth data points are generated by numerically integrating the
    equivalent first-order system with RK4 (see rk4_integrate)."""
    mu = 1.0

    def __init__(self, seed=9):
        super().__init__("EvoPINN10_VanDerPol", [1, 10, 10, 1], seed)

    def _build_data(self):
        r = self.rng
        self.Tr = unif(r, 0, 10, 150)
        self.T0 = np.array([[0.0]])

        def rhs(t, y):
            u, v = y
            return np.array([v, self.mu * (1 - u**2) * v - u])

        ts, ys = rk4_integrate(rhs, [2.0, 0.0], 0.0, 10.0, 2000)
        idx = self.rng.choice(len(ts), size=10, replace=False)
        self.Td = ts[idx].reshape(-1, 1)
        self.ud = ys[idx, 0].reshape(-1, 1)

    def _objectives(self, theta):
        u, d = self.net.forward_with_derivs(theta, self.Tr, {0: 2})
        r = d[(0, 2)] - self.mu * (1 - u**2) * d[(0, 1)] + u
        L_r = mse(r)
        u0 = self.net.forward(theta, self.T0)
        _, d0 = self.net.forward_with_derivs(theta, self.T0, {0: 1})
        L_bc = mse(u0, 2.0)
        L_ic = mse(d0[(0, 1)], 0.0)
        L_d = mse(self.net.forward(theta, self.Td), self.ud)
        return np.array([L_r, L_bc, L_ic, L_d])


class EvoPINN11_LotkaVolterra(EvoPINNProblem):
    """Lotka-Volterra predator-prey ODE system: x' = a*x - b*x*y,
    y' = -c*y + d*x*y, t in [0,20], with 2 network outputs (x and y).
    Data points come from RK4 integration; L_ic also includes an
    energy-conservation term H (the system's known conserved quantity),
    checked against its initial value H0."""
    a, b, c, d_ = 1.1, 0.4, 0.4, 0.1

    def __init__(self, seed=10):
        super().__init__("EvoPINN11_LotkaVolterra", [1, 10, 10, 2], seed)

    def _build_data(self):
        r = self.rng
        self.Tr = unif(r, 0, 20, 200)
        self.T0 = np.array([[0.0]])
        self.y0 = np.array([10.0, 5.0])

        def rhs(t, y):
            x, yy = y
            return np.array([self.a * x - self.b * x * yy,
                              -self.c * yy + self.d_ * x * yy])

        ts, ys = rk4_integrate(rhs, self.y0, 0.0, 20.0, 4000)
        idx = self.rng.choice(len(ts), size=15, replace=False)
        self.Td = ts[idx].reshape(-1, 1)
        self.yd = ys[idx]
        self.H0 = (self.d_ * self.y0[0] - self.c * np.log(self.y0[0])
                   + self.b * self.y0[1] - self.a * np.log(self.y0[1]))

    def _objectives(self, theta):
        u, d = self.net.forward_with_derivs(theta, self.Tr, {0: 1})
        x, y = u[:, :1], u[:, 1:]
        xt, yt = d[(0, 1)][:, :1], d[(0, 1)][:, 1:]
        rx = xt - (self.a * x - self.b * x * y)
        ry = yt - (-self.c * y + self.d_ * x * y)
        L_r = 0.5 * (mse(rx) + mse(ry))
        u0 = self.net.forward(theta, self.T0)[0]
        L_bc = 0.5 * ((u0[0] - self.y0[0])**2 + (u0[1] - self.y0[1])**2)
        x_pos = np.clip(x, 1e-3, None); y_pos = np.clip(y, 1e-3, None)
        H = self.d_ * x_pos - self.c * np.log(x_pos) + self.b * y_pos - self.a * np.log(y_pos)
        L_ic = mse(H, self.H0)
        L_d = mse(self.net.forward(theta, self.Td), self.yd)
        return np.array([L_r, L_bc, L_ic, L_d])


class EvoPINN12_Schrodinger1D(EvoPINNProblem):
    """1D time-dependent Schrodinger equation (harmonic oscillator
    potential V = 0.5*x^2), split into real/imaginary parts u, v (hence 2
    network outputs), x in [-5,5], t in [0,1]. Initial condition is the
    ground-state wavefunction phi0 (ground energy E0=0.5); exact solution
    known in closed form via phi0*cos/sin(E0*t)."""
    E0 = 0.5

    def __init__(self, seed=11):
        super().__init__("EvoPINN12_Schrodinger1D", [2, 12, 12, 2], seed)

    def _build_data(self):
        r = self.rng
        self.Xr = np.hstack([unif(r, -5, 5, 150), unif(r, 0, 1, 150)])
        tb = unif(r, 0, 1, 20)
        self.Xb = np.vstack([np.hstack([-5 * np.ones_like(tb), tb]),
                              np.hstack([5 * np.ones_like(tb), tb])])
        self.ub = np.zeros((self.Xb.shape[0], 2))
        x0 = unif(r, -5, 5, 20)
        self.X0 = np.hstack([x0, np.zeros_like(x0)])
        self.u0 = np.hstack([self._phi0(x0), np.zeros_like(x0)])
        self.Xd = np.hstack([unif(r, -5, 5, 10), unif(r, 0, 1, 10)])
        self.ud = self._exact(self.Xd)

    @staticmethod
    def _phi0(x):
        return np.pi**-0.25 * np.exp(-x**2 / 2)

    def _exact(self, X):
        x, t = X[:, :1], X[:, 1:]
        phi0 = self._phi0(x)
        return np.hstack([phi0 * np.cos(self.E0 * t), -phi0 * np.sin(self.E0 * t)])

    def _objectives(self, theta):
        w, d = self.net.forward_with_derivs(theta, self.Xr, {0: 2, 1: 1})
        u, v = w[:, :1], w[:, 1:]
        ux2, vx2 = d[(0, 2)][:, :1], d[(0, 2)][:, 1:]
        ut, vt = d[(1, 1)][:, :1], d[(1, 1)][:, 1:]
        V = 0.5 * self.Xr[:, :1]**2
        r_real = -vt + 0.5 * ux2 - V * u
        r_imag = ut + 0.5 * vx2 - V * v
        L_r = 0.5 * (mse(r_real) + mse(r_imag))
        L_bc = mse(self.net.forward(theta, self.Xb), self.ub)
        L_ic = mse(self.net.forward(theta, self.X0), self.u0)
        L_d = mse(self.net.forward(theta, self.Xd), self.ud)
        return np.array([L_r, L_bc, L_ic, L_d])


# Registry
_PROBLEM_CLASSES = [
    EvoPINN1_Poisson1D, EvoPINN2_Heat1D, EvoPINN3_Advection1D,
    EvoPINN4_Wave1D, EvoPINN5_Burgers1D, EvoPINN6_AllenCahn1D,
    EvoPINN7_Poisson2D, EvoPINN8_ReactionDiffusion1D,
    EvoPINN9_DampedOscillator, EvoPINN10_VanDerPol,
    EvoPINN11_LotkaVolterra, EvoPINN12_Schrodinger1D,
]

# Instantiate every problem once at import time, keyed by its short name
# (e.g. "EvoPINN1" from "EvoPINN1_Poisson1D") -- this is the shared cache
# consumed by dataset.py's _evopinn_problems().
evopinn_problems = {cls.__name__.split("_", 1)[0]: cls() for cls in _PROBLEM_CLASSES}


def get_evopinn_problem(name):
    """Look up one EvoPINN problem instance by its short name (e.g.
    'EvoPINN3'); falls back to EvoPINN1 if the name isn't recognized."""
    return evopinn_problems.get(name, evopinn_problems['EvoPINN1'])


def get_all_evopinn_problems():
    """Return the full dict of all 12 pre-built EvoPINN problem instances."""
    return evopinn_problems


def create_mixed_evopinn_problems():
    """Alias for get_all_evopinn_problems() (kept for naming compatibility
    with other parts of the codebase/notebooks that may call it by this
    name)."""
    return get_all_evopinn_problems()

    