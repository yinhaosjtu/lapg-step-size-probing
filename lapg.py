"""lapg.py: solvers and model for "Proximal gradient methods with step-size probing for l0-regularized problems".

Contents: the classification model of Section 5 (sigmoid loss with a ridge term), hard thresholding, the
stationarity radius (Proposition 2.1), the solvers IHT (pg_const), SpaRSA (sparsa) and
LAPG / LAPG-P / NLAPG / NLAPG-P (lapg), and a numerical test of
regular (Gamma, sigma)-descent stationarity (descent_check).

The main experiments use the l0 penalty; l1 support is provided only for
the logistic classification reference. Numerical certificates use finite grids
and floating-point tolerances; they are not formal proofs.
"""

import time

import numpy as np
from scipy.special import expit

# ---------------- model ----------------


class ClassificationProblem:
    """f(x) = (1/m) sum_i 1/(1 + exp(b_i a_i^T x)) + (mu/2)||x||^2 for a dense data matrix A (rows a_i^T)."""

    def __init__(self, A, b, mu=1e-3):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        if A.ndim != 2 or min(A.shape) == 0:
            raise ValueError("A must be a nonempty two-dimensional matrix.")
        if b.shape != (A.shape[0],) or not np.all(np.isin(b, (-1.0, 1.0))):
            raise ValueError("b must contain one label in {-1, +1} per row of A.")
        if not np.all(np.isfinite(A)) or not np.isfinite(mu) or mu <= 0:
            raise ValueError("A must be finite and mu must be positive and finite.")
        self.Z = A * b[:, None]
        self.m, self.n = self.Z.shape
        self.mu = float(mu)
        self.counts = dict(feval=0, geval=0)
        # global Lipschitz constant of grad f: |phi''| <= 1/(6 sqrt 3) for the sigmoid loss phi
        self.L = (1.0 / (6.0 * np.sqrt(3.0))) * float(
            np.linalg.norm(self.Z, 2) ** 2
        ) / self.m + self.mu

    def reset_counts(self):
        """Reset objective and gradient evaluation counters."""
        self.counts = dict(feval=0, geval=0)

    def margins(self, x):
        """Return signed classification margins at x."""
        return self.Z @ x

    def value_from_margins(self, t, x):
        """Evaluate the smooth objective using precomputed margins."""
        self.counts["feval"] += 1
        return float(expit(-t).mean() + 0.5 * self.mu * (x @ x))

    def value(self, x):
        """Evaluate the smooth objective and count one function evaluation."""
        return self.value_from_margins(self.margins(x), x)

    def grad_from_margins(self, t, x):
        """Evaluate the smooth gradient using precomputed margins."""
        self.counts["geval"] += 1
        p = expit(-t)
        return (self.Z.T @ (-p * (1.0 - p))) / self.m + self.mu * x


class LogisticProblem(ClassificationProblem):
    """Averaged logistic loss, without a ridge term or an intercept.

    This convex objective is used only with the l1 reference penalty. Labels
    and preprocessing are identical to the nonconvex classification task.
    """

    def __init__(self, A, b):
        super().__init__(A, b)
        self.mu = 0.0
        self.L = float(np.linalg.norm(self.Z, 2) ** 2) / (4.0 * self.m)
        if self.L <= 0:
            raise ValueError("The logistic reference requires a nonzero design matrix.")

    def value_from_margins(self, t, x):
        """Evaluate the smooth objective using precomputed margins."""
        self.counts["feval"] += 1
        return float(np.logaddexp(0.0, -t).mean())

    def grad_from_margins(self, t, x):
        """Evaluate the smooth gradient using precomputed margins."""
        self.counts["geval"] += 1
        return -(self.Z.T @ expit(-t)) / self.m


# ---------------- proximal maps ----------------


def prox_l0(y, gamma, lam):
    """Hard thresholding, a selection of prox_{gamma*lam*||.||_0}(y).

    Entries with |y_i| <= sqrt(2*gamma*lam) are set to zero; ties are resolved to zero.
    """
    z = y.copy()
    z[np.abs(y) <= np.sqrt(2.0 * gamma * lam)] = 0.0
    return z


def g_value(x, lam, kind="l0"):
    """Evaluate the l0 penalty, or the separate logistic reference l1 penalty."""
    if kind == "l1":
        return lam * np.sum(np.abs(x))
    if kind != "l0":
        raise ValueError(kind)
    return lam * np.count_nonzero(x)


def prox(y, gamma, lam, kind="l0"):
    """Apply hard thresholding (l0) or soft thresholding (l1)."""
    if kind == "l1":
        return np.sign(y) * np.maximum(np.abs(y) - gamma * lam, 0.0)
    if kind != "l0":
        raise ValueError(kind)
    return prox_l0(y, gamma, lam)


# ---------------- stationarity radius ----------------


def l0_radius(x, grad, lam):
    """Return the componentwise l0 stationarity-radius bound.

    This is a fixed-point radius only when the gradient vanishes on the
    support of x. At a numerical final point it is a diagnostic, not a
    standalone stationarity certificate. Empty minima are positive infinity.
    """
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be positive and finite.")
    J = x != 0
    a = np.min(x[J] ** 2) / (2.0 * lam) if J.any() else np.inf
    gc = np.abs(grad[~J])
    gc = gc[gc > 0]
    b = np.min(2.0 * lam / gc**2) if gc.size else np.inf
    return float(min(a, b))


# ---------------- solvers ----------------

COLS = ["time", "F", "res", "nnz", "feval", "geval"]


class _Clock:
    """Wall-clock that excludes the time spent on logging."""

    def __init__(self):
        self.acc, self.last = 0.0, time.perf_counter()
        self.running = True
        self.key, self.last_change, self.n_changes = None, 0, -1

    def pause(self):
        if self.running:
            self.acc += time.perf_counter() - self.last
            self.running = False
        return self.acc

    def resume(self):
        if not self.running:
            self.last = time.perf_counter()
            self.running = True


def _validate_solver_options(prob, lam, kind, x0, tol, max_iter, time_limit):
    """Reject invalid common solver parameters before iteration starts."""
    if kind not in ("l0", "l1"):
        raise ValueError("kind must be l0 or l1.")
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(tol) or tol < 0:
        raise ValueError("lam must be positive and tol nonnegative; both must be finite.")
    if max_iter < 0 or int(max_iter) != max_iter or not (0 < time_limit <= np.inf):
        raise ValueError("max_iter must be a nonnegative integer and time_limit positive.")
    if x0 is not None and (np.shape(x0) != (prob.n,) or not np.all(np.isfinite(x0))):
        raise ValueError("x0 must be a finite vector of length prob.n.")


def external_residual(prob, x, grad, lam, kind):
    """Compute L_f times the norm of the step with step size 1/L_f."""
    L = prob.L
    return float(L * np.linalg.norm(x - prox(x - grad / L, 1.0 / L, lam, kind)))


def _record(hist, clock, prob, x, F, grad, lam, kind):
    el = clock.pause()
    key = np.flatnonzero(x).tobytes()
    if key != clock.key:
        clock.key, clock.last_change = key, len(hist)
        clock.n_changes += 1
    res = external_residual(prob, x, grad, lam, kind)
    c = prob.counts
    hist.append((el, F, res, int(np.count_nonzero(x)), c["feval"], c["geval"]))
    clock.resume()
    return res


def _finish(x, hist, clock, extra, prob):
    """Include terminal probing cost in the final observation."""
    final = list(hist[-1])
    final[0] = clock.pause()
    final[4], final[5] = prob.counts["feval"], prob.counts["geval"]
    hist[-1] = tuple(final)
    return dict(
        x=x,
        hist=np.array(hist),
        cols=COLS,
        ident_iter=clock.last_change,
        support_changes=clock.n_changes,
        **extra,
    )


# ----------------------------------------------------------------------------------------------
def probe_grid(gamma_lo, gamma_hi, factor):
    """The fixed finite grid Gamma = {gamma_hi * factor^-j} contained in [gamma_lo, gamma_hi], in decreasing order."""
    if not (0 < gamma_lo <= gamma_hi < np.inf) or not (1 < factor < np.inf):
        raise ValueError("Require 0 < gamma_lo <= gamma_hi and finite factor > 1.")
    J = int(np.floor(np.log(gamma_hi / gamma_lo) / np.log(factor) + 1e-12))
    return gamma_hi * factor ** (-np.arange(J + 1.0))


def lapg(
    prob,
    lam,
    kind,
    x0=None,
    rho=0.5,
    eta=0.5,
    gamma_lo=1e-6,
    gamma_hi=1e6,
    bb=True,
    probe=False,
    probe_every=10,
    probe_factor=4.0,
    sigma=None,
    nm_p=1.0,
    nm_switch=20,
    trace=False,
    tol=1e-6,
    max_iter=20000,
    time_limit=120.0,
):
    """LAPG / LAPG-P / NLAPG / NLAPG-P of the paper.

    probe=True enables step-size probing (period probe_every, grid gamma_hi*probe_factor^-j);
    nm_p<1 enables the averaged nonmonotone line search with switch to monotone acceptance
    after the support has been constant for nm_switch iterations.

    Termination: after every iteration the method checks the external residual L_f||x-P_{1/L_f}(x)|| and its own
    step residual ||d^k||/gamma_k. If one of them is <= tol, a method without probing stops. A method with probing
    first probes all step sizes in Gamma at the current point and stops only if no probing point is accepted;
    otherwise it moves to the best accepted point and continues. Hence every final point of a method with probing
    passes the implemented probing test if termination is successful. Reaching
    max_iter or time_limit does not imply stationarity or certification. At least
    one iteration is performed when the iteration and time budgets permit it,
    also from a warm start.
    """
    _validate_solver_options(prob, lam, kind, x0, tol, max_iter, time_limit)
    if not (0 < rho < 1 and 0 < eta < 1 and 0 < nm_p <= 1):
        raise ValueError("Require rho, eta in (0, 1), and nm_p in (0, 1].")
    if not (0 < gamma_lo <= gamma_hi < np.inf):
        raise ValueError("Require finite 0 < gamma_lo <= gamma_hi.")
    if probe_every < 1 or int(probe_every) != probe_every or nm_switch < 0:
        raise ValueError("probe_every must be a positive integer; nm_switch must be nonnegative.")
    if sigma is not None and not (0 < sigma <= 1 - rho):
        raise ValueError("Require 0 < sigma <= 1 - rho.")
    prob.reset_counts()
    n = prob.n
    x = np.zeros(n) if x0 is None else x0.copy()
    clock = _Clock()
    t = prob.margins(x)
    f = prob.value_from_margins(t, x)
    g = prob.grad_from_margins(t, x)
    F = f + g_value(x, lam, kind)
    hist, gam_log, trial_log = [], [], []
    res = _record(hist, clock, prob, x, F, g, lam, kind)
    X = [x.copy()] if trace else None
    x_prev = g_prev = None
    gamma_prev = None
    n_probe = n_probe_moves = n_final_probes = 0
    sig = 0.5 * (1.0 - rho) if sigma is None else sigma
    Phi = F
    p_eff, nm_key, nm_stable, switch_iter = nm_p, np.flatnonzero(x).tobytes(), 0, -1
    Gamma = probe_grid(gamma_lo, gamma_hi, probe_factor) if probe else None
    sd_viol = -np.inf
    want_stop, converged = False, False
    k = 0
    while k < max_iter and clock.pause() < time_limit:
        clock.resume()
        if want_stop and not probe:
            converged = True
            break
        final_probe = want_stop  # Run probing after a residual-based stopping candidate.
        ref = Phi if p_eff < 1.0 else F  # reference value of the acceptance tests
        produced = []  # (step size, point) pairs, to determine gamma_k
        trials = 0
        if not final_probe:
            # --- trial step size
            if bb and x_prev is not None:
                s, y = x - x_prev, g - g_prev
                sy = abs(s @ y)
                gam0 = (s @ s) / sy if sy > 0 else gamma_hi
            elif (not bb) and gamma_prev is not None:
                gam0 = gamma_prev
            else:
                gam0 = 1.0
            gamma = float(np.clip(gam0, gamma_lo, gamma_hi))
            # Backtrack against the smooth model or the averaged objective reference.
            trials = 1
            while True:
                xn = prox(x - gamma * g, gamma, lam, kind)
                d = xn - x
                tn = prob.margins(xn)
                fn = prob.value_from_margins(tn, xn)
                if p_eff < 1.0:
                    if fn + g_value(xn, lam, kind) <= Phi - sig / (2.0 * gamma) * (
                        d @ d
                    ) + 1e-14 * max(1.0, abs(Phi)):
                        break
                elif fn <= f + g @ d + rho / (2.0 * gamma) * (d @ d) + 1e-14 * max(1.0, abs(f)):
                    break
                gamma *= eta
                trials += 1
                if gamma < 1e-30:
                    raise RuntimeError("Backtracking failed: step size underflow.")
            Fn = fn + g_value(xn, lam, kind)
            produced.append((gamma, xn))
            bestF = Fn
        else:
            xn, tn, fn, Fn, d, gamma = x, t, f, F, np.zeros(n), np.inf
            bestF = F  # The stopping probe must improve the current objective.
        moved_by_probe = False
        if probe and (final_probe or k % probe_every == 0):
            n_probe += 1
            n_final_probes += int(final_probe)
            for gp in Gamma:  # decreasing order
                u = prox(x - gp * g, gp, lam, kind)
                du = u - x
                produced.append((gp, u))
                if not np.any(du):
                    continue
                tu = prob.margins(u)
                fu = prob.value_from_margins(tu, u)
                Fu = fu + g_value(u, lam, kind)
                # probe acceptance: sufficient decrease of F with respect to the reference value (Algorithm 1)
                if Fu <= ref - sig / (2.0 * gp) * (du @ du) and Fu < bestF:
                    bestF, xn, tn, fn, Fn, d = Fu, u, tu, fu, Fu, du
                    moved_by_probe = True
            if moved_by_probe:
                n_probe_moves += 1
        if final_probe and not moved_by_probe:
            converged = True
            break
        # gamma_k: the largest step size that produced x^{k+1} (line search point and/or probing points)
        gamma = max(gp for gp, u in produced if np.array_equal(u, xn))
        step_res = float(np.sqrt(d @ d) / gamma)
        gn = prob.grad_from_margins(tn, xn)
        sd_viol = max(sd_viol, (Fn - ref + sig / (2.0 * gamma) * (d @ d)) / max(1.0, abs(F)))
        Phi = (1.0 - p_eff) * Phi + p_eff * Fn
        if (
            p_eff < 1.0
        ):  # switch to monotone acceptance once the support has settled for nm_switch steps
            key = np.flatnonzero(xn).tobytes()
            nm_stable = nm_stable + 1 if key == nm_key else 0
            nm_key = key
            if nm_stable >= nm_switch:
                p_eff, switch_iter = 1.0, k
        gamma_prev = gamma
        gam_log.append(gamma)
        trial_log.append(trials)
        x_prev, g_prev = x, g
        x, t, f, g, F = xn, tn, fn, gn, Fn
        k += 1
        res = _record(hist, clock, prob, x, F, g, lam, kind)
        if trace:
            X.append(x.copy())
        want_stop = (res <= tol) or (step_res <= tol)
    converged = converged or (want_stop and not probe)
    return _finish(
        x,
        hist,
        clock,
        dict(
            iters=k,
            converged=converged,
            gammas=np.array(gam_log),
            trials=np.array(trial_log),
            n_probe=n_probe,
            n_probe_moves=n_probe_moves,
            n_final_probes=n_final_probes,
            switch_iter=switch_iter,
            X=X,
            sd_violation=sd_viol,
        ),
        prob,
    )


# ----------------------------------------------------------------------------------------------
def pg_const(prob, lam, kind, x0=None, tol=1e-6, max_iter=200000, time_limit=120.0):
    """Proximal gradient with the constant step 1/L (IHT when kind == 'l0')."""
    _validate_solver_options(prob, lam, kind, x0, tol, max_iter, time_limit)
    prob.reset_counts()
    x = np.zeros(prob.n) if x0 is None else x0.copy()
    clock = _Clock()
    t = prob.margins(x)
    f = prob.value_from_margins(t, x)
    g = prob.grad_from_margins(t, x)
    hist = []
    res = _record(hist, clock, prob, x, f + g_value(x, lam, kind), g, lam, kind)
    k, gamma = 0, 1.0 / prob.L
    while res > tol and k < max_iter and clock.pause() < time_limit:
        clock.resume()
        x = prox(x - gamma * g, gamma, lam, kind)
        t = prob.margins(x)
        f = prob.value_from_margins(t, x)
        g = prob.grad_from_margins(t, x)
        k += 1
        res = _record(hist, clock, prob, x, f + g_value(x, lam, kind), g, lam, kind)
    return _finish(x, hist, clock, dict(iters=k, converged=res <= tol), prob)


def sparsa(
    prob,
    lam,
    kind,
    x0=None,
    M=5,
    sigma=1e-2,
    eta=0.5,
    gamma_lo=1e-6,
    gamma_hi=1e6,
    tol=1e-6,
    max_iter=20000,
    time_limit=120.0,
):
    """Run the SpaRSA-type baseline with an absolute-curvature BB step.

    This is the included research implementation, not the authors' official
    SpaRSA software. Termination uses the external proximal-gradient residual
    or the accepted-step residual, matching the manuscript stopping rule.
    """
    if M < 1 or int(M) != M or not (0 < sigma < 1 and 0 < eta < 1):
        raise ValueError("M must be a positive integer; sigma and eta must lie in (0, 1).")
    if not (0 < gamma_lo <= gamma_hi < np.inf):
        raise ValueError("Require finite 0 < gamma_lo <= gamma_hi.")
    _validate_solver_options(prob, lam, kind, x0, tol, max_iter, time_limit)
    prob.reset_counts()
    x = np.zeros(prob.n) if x0 is None else x0.copy()
    clock = _Clock()
    t = prob.margins(x)
    f = prob.value_from_margins(t, x)
    g = prob.grad_from_margins(t, x)
    F = f + g_value(x, lam, kind)
    hist = []
    res = _record(hist, clock, prob, x, F, g, lam, kind)
    Fhist = [F]
    x_prev = g_prev = None
    k = 0
    step_res = np.inf
    while (
        (k == 0 or (res > tol and step_res > tol)) and k < max_iter and clock.pause() < time_limit
    ):
        clock.resume()
        if x_prev is not None:
            s, y = x - x_prev, g - g_prev
            sy = abs(s @ y)
            gam0 = (s @ s) / sy if sy > 0 else gamma_hi
        else:
            gam0 = 1.0
        gamma = float(np.clip(gam0, gamma_lo, gamma_hi))
        Fref = max(Fhist[-M:])
        while True:
            xn = prox(x - gamma * g, gamma, lam, kind)
            d = xn - x
            tn = prob.margins(xn)
            fn = prob.value_from_margins(tn, xn)
            Fn = fn + g_value(xn, lam, kind)
            if Fn <= Fref - sigma / (2.0 * gamma) * (d @ d) + 1e-14 * max(1.0, abs(Fref)):
                break
            gamma *= eta
            if gamma < 1e-30:
                raise RuntimeError("SpaRSA backtracking failed: step size underflow.")
        step_res = float(np.linalg.norm(d) / gamma)
        gn = prob.grad_from_margins(tn, xn)
        x_prev, g_prev = x, g
        x, t, f, g, F = xn, tn, fn, gn, Fn
        Fhist.append(F)
        k += 1
        res = _record(hist, clock, prob, x, F, g, lam, kind)
    return _finish(x, hist, clock, dict(iters=k, converged=(res <= tol or step_res <= tol)), prob)


# ---------------- regular (Gamma, sigma)-descent stationarity of a point ----------------


def descent_check(prob, x, lam, sigma=0.25, Gamma=None):
    """Checks regular (Gamma, sigma)-descent stationarity of x for g = lam*||.||_0.

    This is a floating-point diagnostic on a finite grid, not a proof. It may
    be called on any point, independently of the solver termination status.

    Returns (certified, largest relative decrease (F(x)-F(u))/F(x) over u = P_gamma(x), gamma in Gamma_reg(x)).
    Step sizes with a tie |x_i - gamma*grad_i| = sqrt(2*gamma*lam) are skipped, since they are not in Gamma_reg(x).
    """
    Gamma = probe_grid(1e-6, 1e6, 4.0) if Gamma is None else Gamma
    g = prob.grad_from_margins(prob.margins(x), x)
    Fx = prob.value(x) + lam * np.count_nonzero(x)
    ok, best = True, 0.0
    for gm in Gamma:
        y = x - gm * g
        thr = np.sqrt(2 * gm * lam)
        if np.min(np.abs(np.abs(y) - thr)) <= 1e-12 * thr:
            continue
        u = prox(y, gm, lam, "l0")
        Fu = prob.value(u) + lam * np.count_nonzero(u)
        best = max(best, (Fx - Fu) / abs(Fx))
        if Fu < Fx - sigma / (2 * gm) * ((u - x) @ (u - x)) - 1e-9 * max(1.0, abs(Fx)):
            ok = False
    return ok, best
