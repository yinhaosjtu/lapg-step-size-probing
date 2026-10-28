"""Run numerical experiments for step-size probing with the l0 penalty.

Use ``python experiments.py --help`` for available commands. Result directories
are selected with ``--results-dir``. The objective reference is the smallest
final value of the six compared methods on the same instance.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

import lapg as L

RES = str(Path(__file__).resolve().parent / "results")
METHODS = ["IHT", "SpaRSA", "LAPG", "LAPG-P", "NLAPG", "NLAPG-P"]


def index_range(value):
    """Parse a nonempty, half-open nonnegative index range, e.g. 0:20."""
    try:
        start, stop = map(int, value.split(":"))
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("Use START:STOP, for example 0:20.") from exc
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("Require 0 <= START < STOP.")
    return value


def _write_csv(rows, filename):
    """Publish a CSV atomically so interrupted writes are not mistaken for results."""
    destination = Path(RES) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".tmp", dir=destination.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            pd.DataFrame(rows).to_csv(stream, index=False)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)


def _save_arrays(filename, **arrays):
    """Atomically save numerical arrays without Python pickle objects."""
    directory = Path(RES) / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(directory / filename)


def _write_json(data, filename):
    """Write a small JSON result using an explicit UTF-8 encoding."""
    with (Path(RES) / filename).open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def _run_metadata(args):
    """Record the invocation and installed dependency versions beside its results."""
    options = {key: value for key, value in vars(args).items() if key != "fn"}
    return {
        "command": sys.argv,
        "options": options,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib")
        },
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
            for name in ("lapg.py", "experiments.py", "report.py")
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def lam0(P):
    """lambda_0 = ||grad f(0)||_inf^2 / (2 L_f): the origin is 1/L_f-stationary iff lambda >= lambda_0."""
    x0 = np.zeros(P.n)
    return float(np.abs(P.grad_from_margins(P.margins(x0), x0)).max() ** 2 / (2.0 * P.L))


def solvers(P, lam, iht_iters=2_000_000, iht_time_limit=np.inf):
    """Return the six manuscript methods, all initialized at zero.

    The 5-second IHT budget is used only for real-data experiments. Other
    methods have no wall-clock limit; iteration caps are safeguards whose
    exhaustion must be checked before treating results as converged.
    """
    return {
        "IHT": lambda: L.pg_const(P, lam, "l0", max_iter=iht_iters, time_limit=iht_time_limit),
        "SpaRSA": lambda: L.sparsa(P, lam, "l0", max_iter=200_000, time_limit=np.inf),
        "LAPG": lambda: L.lapg(P, lam, "l0", max_iter=200_000, time_limit=np.inf),
        "LAPG-P": lambda: L.lapg(P, lam, "l0", probe=True, max_iter=200_000, time_limit=np.inf),
        "NLAPG": lambda: L.lapg(P, lam, "l0", nm_p=0.2, max_iter=200_000, time_limit=np.inf),
        "NLAPG-P": lambda: L.lapg(
            P, lam, "l0", nm_p=0.2, probe=True, max_iter=200_000, time_limit=np.inf
        ),
    }


def Fval(P, x, lam):
    """Evaluate the sigmoid objective including its l0 penalty."""
    return P.value(x) + lam * np.count_nonzero(x)


def run_instance(
    P, lam, iht_iters=2_000_000, extra=None, iht_time_limit=np.inf, archive_name=None
):
    """Runs all methods on one instance; returns rows (one per method) with F, certification and F_best."""
    rows, points = [], {}
    for name, run in solvers(P, lam, iht_iters, iht_time_limit).items():
        r = run()
        x, h = r["x"], r["hist"]
        g = P.grad_from_margins(P.margins(x), x)
        ok, mx = L.descent_check(P, x, lam)
        row = dict(
            method=name,
            F=h[-1, 1],
            nnz=int(h[-1, 3]),
            iters=r["iters"],
            time=h[-1, 0],
            feval=int(h[-1, 4]),
            conv=r["converged"],
            residual=h[-1, 2],
            geval=int(h[-1, 5]),
            radiusL=L.l0_radius(x, g, lam) * P.L,
            changes=r["support_changes"],
            certified=ok,
            max_rel_decrease=mx,
        )
        if extra is not None:
            row.update(extra(x))
        rows.append(row)
        points[name] = x
    if archive_name is not None:
        _save_arrays(archive_name, **points, lam=np.asarray(lam))
    Fref = min(r["F"] for r in rows)
    for r in rows:
        r["Fref"] = Fref
    return rows


def synthetic_arrays(seed, m, n, spread=0.0):
    """Generate the actual synthetic inputs and planted coefficient vector.

    The local random generators make the inputs independent of call order.
    Labels use the unscaled Gaussian features; optional column rescaling is
    then applied to the observed design.
    """
    if m < 1 or n < 20 or not np.isfinite(spread) or spread < 0:
        raise ValueError("Require m >= 1, n >= 20, and finite spread >= 0.")
    rng = np.random.default_rng(1000 + seed)
    A = rng.standard_normal((m, n))
    xt = np.zeros(n)
    support = rng.choice(n, 20, replace=False)
    xt[support] = rng.choice([-1, 1], 20) * (1 + rng.random(20))
    b = np.sign(A @ xt + 0.5 * rng.standard_normal(m))
    b[b == 0] = 1
    if spread > 0:
        A = A * 10 ** np.random.default_rng(3000 + seed).uniform(-spread, spread, n)
    return A, b, xt


def synth_problem(seed, m, n, spread=0.0):
    """Build a classification objective and return its planted coefficients."""
    A, b, xt = synthetic_arrays(seed, m, n, spread)
    return L.ClassificationProblem(A, b, mu=1e-3), xt


# ------------------------------------------------------------------ Example 2.1
def example(_):
    """Compute the one-dimensional illustration and probing points."""
    mu, lam, rho, sig = 0.005, 1.0, 0.5, 0.25
    f = lambda x: -x + 0.5 * np.log1p(x * x) + 0.5 * mu * x * x
    df = lambda x: -1.0 + x / (1.0 + x * x) + mu * x
    F = lambda x: f(x) + lam * (x != 0)
    H = lambda y, gm: y if abs(y) > np.sqrt(2 * gm * lam) else 0.0
    Lf = 1.0 + mu

    def method(probe, x=0.0, iters=500, Gamma=tuple(4.0**j for j in range(8))):
        path = [x]
        for _ in range(iters):
            gm = 1.0
            while True:
                z = H(x - gm * df(x), gm)
                if f(z) <= f(x) + df(x) * (z - x) + rho / (2 * gm) * (z - x) ** 2:
                    break
                gm *= 0.5
            if probe:
                for g2 in Gamma:
                    u = H(x - g2 * df(x), g2)
                    if u != x and F(u) <= F(x) - sig / (2 * g2) * (u - x) ** 2 and F(u) < F(z):
                        z = u
            x = z
            path.append(x)
        return x, path

    xs = np.linspace(1, 400, 400001)
    xg = xs[np.argmin(F(xs))]
    xi = 0.0
    for _ in range(500):
        xi = H(xi - df(xi) / Lf, 1 / Lf)
    xl, _ = method(False)
    xp, path = method(True)
    out = dict(
        x_global=xg,
        F_global=F(xg),
        iht=xi,
        lapg=xl,
        lapg_p=xp,
        F_lapg_p=F(xp),
        path=path[:20],
        probes={
            str(g): [H(-g * df(0.0), g), F(H(-g * df(0.0), g))]
            for g in [1 / Lf, 2, 4, 16, 64, 256, 1024]
        },
    )
    _write_json(out, "example1d.json")
    print(f"x*={xg:.3f} F(x*)={F(xg):.4f} | IHT->{xi} LAPG->{xl} LAPG-P->{xp:.3f} F={F(xp):.4f}")


# ------------------------------------------------------------------ Synthetic data
def synth(a):
    """Run six methods on the requested half-open range of Gaussian instances."""
    s0, s1 = map(int, a.seeds.split(":"))
    rows = []
    for seed in range(s0, s1):
        A, b, xt = synthetic_arrays(seed, a.m, a.n)
        P = L.ClassificationProblem(A, b)
        if a.save_inputs:
            _save_arrays(
                f"synthetic_{a.m}x{a.n}_seed{seed}.npz", A=A, b=b, x_true=xt, seed=np.asarray(seed)
            )
        lam = 0.8 * lam0(P)
        tp = lambda x: dict(tp=len(set(np.flatnonzero(x)) & set(np.flatnonzero(xt))))
        archive = f"solutions_synthetic_{a.m}x{a.n}_seed{seed}.npz" if a.save_inputs else None
        for r in run_instance(P, lam, extra=tp, archive_name=archive):
            rows.append(dict(m=a.m, n=a.n, seed=seed, **r))
        print("synth", a.m, a.n, seed, flush=True)
    _write_csv(rows, f"synth_{a.m}x{a.n}_{s0}_{s1}.csv")


# ------------------------------------------------------------------ Real data with label noise
def real_arrays(data, rep, noise):
    """Return processed arrays and the exact split/noise metadata.

    Quadratic features are constructed before splitting, but constant-column
    selection and standardization are fitted on the training partition only.
    Labels are +1 for benign breast samples and even digits, respectively.
    """
    from sklearn.datasets import load_breast_cancer, load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    if data not in ("breast_cancer", "digits_parity"):
        raise ValueError("Unknown data set; choose breast_cancer or digits_parity.")
    if not np.isfinite(noise) or not 0 <= noise <= 1 or rep < 0:
        raise ValueError("Require noise in [0, 1] and rep >= 0.")
    if data == "breast_cancer":
        dataset = load_breast_cancer()
        y = np.where(dataset.target == 1, 1.0, -1.0)
    else:
        dataset = load_digits()
        y = np.where(dataset.target % 2 == 0, 1.0, -1.0)
    X = PolynomialFeatures(2, include_bias=False).fit_transform(dataset.data)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=rep)
    keep = X[tr].std(0) > 1e-12
    scaler = StandardScaler().fit(X[tr][:, keep])
    Xtr = scaler.transform(X[tr][:, keep])
    Xte = scaler.transform(X[te][:, keep])
    ytr, yte = y[tr].copy(), y[te]
    flip = np.random.default_rng(500 + rep).random(len(ytr)) < noise
    ytr[flip] *= -1
    return {
        "X_train": Xtr,
        "X_test": Xte,
        "y_train": ytr,
        "y_test": yte,
        "train_indices": tr,
        "test_indices": te,
        "flipped_train_mask": flip,
        "feature_mask": keep,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }


def real_problem(data, rep, noise):
    """Build a training objective and return clean held-out test arrays."""
    arrays = real_arrays(data, rep, noise)
    return (
        L.ClassificationProblem(arrays["X_train"], arrays["y_train"], mu=1e-3),
        arrays["X_test"],
        arrays["y_test"],
    )


def real(a):
    """Run real-data instances and the separate convex accuracy reference."""
    r0, r1 = map(int, a.reps.split(":"))
    rows = []
    for rep in range(r0, r1):
        arrays = real_arrays(a.data, rep, a.noise)
        P = L.ClassificationProblem(arrays["X_train"], arrays["y_train"])
        Xte, yte = arrays["X_test"], arrays["y_test"]
        if a.save_inputs:
            _save_arrays(
                f"real_{a.data}_noise{a.noise}_split{rep}.npz",
                **arrays,
                rep=np.asarray(rep),
                noise=np.asarray(a.noise),
            )
        acc = lambda x: dict(acc=float(np.mean(np.where(Xte @ x >= 0, 1.0, -1.0) == yte)))
        baseline = L.LogisticProblem(P.Z, np.ones(P.m))
        gradient0 = baseline.grad_from_margins(np.zeros(P.m), np.zeros(P.n))
        lambda1 = 0.05 * float(np.max(np.abs(gradient0)))
        logistic = L.sparsa(baseline, lambda1, "l1", max_iter=200_000, time_limit=np.inf)
        logistic_accuracy = acc(logistic["x"])["acc"]
        l0 = lam0(P)
        if a.save_inputs:
            _save_arrays(
                f"logistic_{a.data}_noise{a.noise}_split{rep}.npz",
                x=logistic["x"],
                lam=np.asarray(lambda1),
            )
        for c in [0.25, 0.5]:
            archive = (
                f"solutions_{a.data}_noise{a.noise}_split{rep}_lambda{c}.npz"
                if a.save_inputs
                else None
            )
            for r in run_instance(P, c * l0, extra=acc, iht_time_limit=5.0, archive_name=archive):
                r.update(
                    logistic_acc=logistic_accuracy,
                    logistic_nnz=int(np.count_nonzero(logistic["x"])),
                    logistic_converged=logistic["converged"],
                    logistic_lambda=lambda1,
                )
                rows.append(dict(data=a.data, noise=a.noise, rep=rep, lam_rel=c, **r))
        print(a.data, a.noise, rep, "done", flush=True)
    _write_csv(rows, f"real_{a.data}_{a.noise}_{r0}_{r1}.csv")


# ------------------------------------------------------------------ Parameter sensitivity
ABL = {
    "base": dict(),
    "T=1": dict(probe_every=1),
    "T=50": dict(probe_every=50),
    "grid 2": dict(probe_factor=2.0),
    "grid 16": dict(probe_factor=16.0),
    "p=0.5": dict(nm_p=0.5),
    "p=0.9": dict(nm_p=0.9),
    "no probing": dict(probe=False),
}


def ablation(a):
    """Run eight variants on ten synthetic and twenty breast-cancer cases."""
    probs = []
    for seed in range(10):
        P, _ = synth_problem(seed, 500, 2000)
        probs.append(("synthetic", f"s{seed}", P, 0.8 * lam0(P)))
    for noise in [0.0, 0.2]:
        for rep in range(10):
            P, _, _ = real_problem("breast_cancer", rep, noise)
            probs.append((f"breast cancer p={noise}", f"r{rep}", P, 0.5 * lam0(P)))
    rows = []
    for grp, inst, P, lam in probs:
        for name, kw in ABL.items():
            opts = dict(probe=True, probe_every=10, probe_factor=4.0, nm_p=0.2)
            opts.update(kw)
            r = L.lapg(P, lam, "l0", max_iter=200_000, time_limit=np.inf, **opts)
            if a.save_inputs:
                _save_arrays(
                    f"ablation_{grp.replace(' ', '_')}_{inst}_{name.replace(' ', '_')}.npz",
                    x=r["x"],
                    lam=np.asarray(lam),
                )
            rows.append(
                dict(
                    group=grp,
                    inst=inst,
                    variant=name,
                    F=r["hist"][-1, 1],
                    iters=r["iters"],
                    feval=int(r["hist"][-1, 4]),
                    time=r["hist"][-1, 0],
                    conv=r["converged"],
                )
            )
        print(grp, inst, "done", flush=True)
    _write_csv(rows, "ablation.csv")


# ------------------------------------------------------------------ descent profile of final points (figure)
def profile(a):
    """Evaluate excess decrease at one synthetic and one breast-cancer instance.

    The real-data panel is chosen deterministically as the first split/noise/
    penalty setting in the manuscript's certification subset where LAPG fails
    the numerical certification check. If none fails, record that fact rather
    than claiming that the panel satisfies the manuscript's selection rule.
    """
    P1, _ = synth_problem(0, 500, 2000)
    if a.save_inputs:
        A, b, planted = synthetic_arrays(0, 500, 2000)
        _save_arrays("profile_synthetic_inputs.npz", A=A, b=b, x_true=planted)
    selected = None
    for rep in range(10):
        for noise in (0.0, 0.2):
            problem, _, _ = real_problem("breast_cancer", rep, noise)
            for relative_lambda in (0.25, 0.5):
                lam = relative_lambda * lam0(problem)
                run = solvers(problem, lam)["LAPG"]()
                if not L.descent_check(problem, run["x"], lam)[0]:
                    selected = (problem, lam, rep, noise, relative_lambda)
                    break
            if selected is not None:
                break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError(
            "No uncertified LAPG point was found in the 40-instance breast-cancer subset. Inspect results and revise the figure selection explicitly."
        )
    P2, lambda2, rep, noise, relative_lambda = selected
    if a.save_inputs:
        _save_arrays(
            "profile_breast_cancer_inputs.npz", **real_arrays("breast_cancer", rep, noise)
        )
    out = {}
    for title, problem, lam, iht_seconds in (
        ("synthetic", P1, 0.8 * lam0(P1), np.inf),
        ("breast cancer", P2, lambda2, 5.0),
    ):
        grid = np.logspace(-1, 6, 500) / problem.L
        result = {
            "L": problem.L,
            "Gamma": list(L.probe_grid(1e-6, 1e6, 4.0)),
            "selection": (
                {"seed": 0}
                if title == "synthetic"
                else {"rep": rep, "noise": noise, "lam_rel": relative_lambda}
            ),
        }
        for method, run in solvers(problem, lam, iht_time_limit=iht_seconds).items():
            final = run()
            x = final["x"]
            if a.save_inputs:
                _save_arrays(
                    f"profile_{title.replace(' ', '_')}_{method}.npz", x=x, lam=np.asarray(lam)
                )
            gradient = problem.grad_from_margins(problem.margins(x), x)
            objective = Fval(problem, x, lam)

            def excess(gamma):
                point = L.prox(x - gamma * gradient, gamma, lam, "l0")
                return float(
                    (
                        objective
                        - Fval(problem, point, lam)
                        - 0.25 / (2 * gamma) * np.sum((point - x) ** 2)
                    )
                    / objective
                )

            result[method] = {
                "gammaL": list(grid * problem.L),
                "excess": [excess(gamma) for gamma in grid],
                "grid_gammaL": list(np.asarray(result["Gamma"]) * problem.L),
                "grid_excess": [excess(gamma) for gamma in result["Gamma"]],
                "radiusL": L.l0_radius(x, gradient, lam) * problem.L,
                "F": objective,
                "converged": final["converged"],
            }
        out[title] = result
        print(title, "done", flush=True)
    _write_json(out, "profile.json")


# ------------------------------------------------------------------ verification of the theory
def verify(a):
    """Symbolic checks (SymPy), rigorous interval checks (mpmath.iv) and randomized property tests."""
    import sympy as sp
    from mpmath import iv, mp, mpf
    from mpmath import log as mlog

    out, ok_all = [], True

    def rep(name, ok, info=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        out.append(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")
        print(out[-1], flush=True)

    # S1 proximal-step bound: d^2/(2g) <= H + G d  =>  d <= gG + sqrt(g^2G^2+2gH) <= 2gG + sqrt(2gH)
    d, g, G, H, Lc, s_, r, lam, m, t, gmin, gmax, Phi, Phin, p = sp.symbols(
        "d gamma G H L sigma rho lambda m t gmin gmax Phi Phin p", positive=True
    )
    q = d**2 - 2 * g * G * d - 2 * g * H
    rp = g * G + sp.sqrt(g**2 * G**2 + 2 * g * H)
    rep("S1a larger root of the quadratic", sp.simplify(q.subs(d, rp)) == 0)
    lhs, rhs = rp, 2 * g * G + sp.sqrt(2 * g * H)
    diff_sq = sp.expand(
        (rhs - g * G) ** 2 - (g**2 * G**2 + 2 * g * H)
    )  # (gG+sqrt(2gH))^2 - (g^2G^2+2gH) = 2gG sqrt(2gH) >= 0
    rep(
        "S1b root <= 2gG+sqrt(2gH)",
        sp.simplify(diff_sq - 2 * g * G * sp.sqrt(2 * g * H)) == 0,
        "(difference of squares = 2 gamma G sqrt(2 gamma H) >= 0)",
    )
    # S2 descent-stationarity bound: 1/(2g)-L/2-s/(2g) = (1-s-gL)/(2g)
    rep(
        "S2 descent-stationarity bound equivalence",
        sp.simplify(1 / (2 * g) - Lc / 2 - s_ / (2 * g) - (1 - s_ - g * Lc) / (2 * g)) == 0,
    )
    # S3 sufficient-decrease inequality: adding (2.1) and (3.1)
    fx, fz, gx, gz, ip, nd = sp.symbols("fx fz gx gz ip nd2", real=True)
    sum_ = (fx + ip + r / (2 * g) * nd) + (gx - ip - nd / (2 * g))  # upper bound for f(z)+g(z)
    rep(
        "S3 sufficient-decrease inequality F(z) <= F(x) - (1-rho)/(2g)||d||^2",
        sp.simplify(sum_ - (fx + gx - (1 - r) / (2 * g) * nd)) == 0,
    )
    # S4 support-entry decrease: decrease per entering index
    rep(
        "S4a entering index: sigma/(2g)*2g*lam = sigma*lam",
        sp.simplify(s_ / (2 * g) * 2 * g * lam - s_ * lam) == 0,
    )
    # S5 local convergence inequality: min_w <h,w> + m/2 w^2 = -h^2/(2m); KL with chi(t)=sqrt(2t/m)
    w, h = sp.symbols("w h", real=True)
    wmin = sp.solve(sp.diff(h * w + m / 2 * w**2, w), w)[0]
    rep(
        "S5a strong convexity bound",
        sp.simplify((h * wmin + m / 2 * wmin**2) + h**2 / (2 * m)) == 0,
    )
    chi = sp.sqrt(2 * t / m)
    rep(
        "S5b chi'(t)*sqrt(2 m t) = 1",
        sp.simplify(sp.diff(chi, t) * sp.sqrt(2 * m * t) - 1) == 0,
        "(so chi'(psi-psi*)||grad psi|| >= 1 when psi-psi* <= ||grad psi||^2/(2m))",
    )
    # S6 nonmonotone averaging identity: F(x^{k+1}) = Phi_k + (Phi_{k+1}-Phi_k)/p_k
    Fn = sp.Symbol("Fn", real=True)
    rep(
        "S6 nonmonotone identity",
        sp.simplify(Phi + (((1 - p) * Phi + p * Fn) - Phi) / p - Fn) == 0,
    )
    # S7 tie equation (x - g c)^2 = 2 g lam has at most two positive roots in g
    xx, cc = sp.symbols("x c", real=True)
    poly = sp.Poly(sp.expand((xx - g * cc) ** 2 - 2 * g * lam), g)
    rep(
        "S7 tie equation is quadratic (c!=0) or linear with slope -2 lam (c=0)",
        poly.degree() == 2
        and sp.simplify(poly.subs(cc, 0).as_expr() - (xx**2 - 2 * g * lam)) == 0,
    )
    # S8 one-dimensional example
    X = sp.Symbol("X", real=True)
    mu = sp.Rational(1, 200)
    f = -X + sp.log(1 + X**2) / 2 + mu / 2 * X**2
    f2 = sp.simplify(sp.diff(f, X, 2) - mu)
    rep("S8a f'' = (1-x^2)/(1+x^2)^2 + mu", sp.simplify(f2 - (1 - X**2) / (1 + X**2) ** 2) == 0)
    crit = sp.solve(sp.diff(f2, X), X)
    vals = sorted([f2.subs(X, c) for c in crit] + [sp.limit(f2, X, sp.oo)])
    rep(
        "S8b range of (1-x^2)/(1+x^2)^2 is [-1/8, 1]",
        vals[0] == sp.Rational(-1, 8) and vals[-1] == 1,
        f"(critical values {vals})",
    )
    rep("S8c f'(0) = -1, gamma_hat(0) = 2 lam/f'(0)^2 = 2", sp.diff(f, X).subs(X, 0) == -1)
    Fi = (
        lambda x: -x + iv.log(1 + x * x) / 2 + iv.mpf(1) / 400 * x * x + 1
    )  # F(x) for x != 0, interval arithmetic
    iv.dps = 30
    F16 = Fi(iv.mpf(16))
    rep(
        "S8d F(16) in",
        F16.b < -11.58 and F16.a > -11.59,
        f"[{F16.a}, {F16.b}] < -8 = inf over sigma<1 of -8 sigma",
    )
    for gm in [4, 16, 64, 256]:
        Fu = Fi(iv.mpf(gm))
        bound = -0.25 / (2 * gm) * gm * gm
        rep(
            f"S8e probe gamma={gm}: F(P_g(0)) <= F(0) - (1/4)/(2g)|u|^2",
            Fu.b <= bound,
            f"(F in [{float(Fu.a):.4f},{float(Fu.b):.4f}], bound {bound})",
        )
    rep("S8f gamma=256 fails for sigma>0.67", Fi(iv.mpf(256)).a > -0.67 * 128)
    fiv = lambda x: -x + iv.log(1 + x * x) / 2 + iv.mpf(1) / 400 * x * x
    t4 = fiv(iv.mpf(4)) + 4
    t12 = fiv(iv.mpf(12)) + 12  # f(u)-f(0)-f'(0)u with f(0)=0, f'(0)=-1
    rep(
        "S8g test (3.1), rho=1/2: fails at gamma=4, holds at gamma=12",
        t4.a > 0.25 / 4 * 16 and t12.b < 0.25 / 12 * 144,
        f"(gamma=4: {float(t4.a):.3f} > 1.0; gamma=12: {float(t12.b):.3f} < 3.0)",
    )
    cub = sp.Poly(sp.cancel(sp.diff(f, X) * (1 + X**2)), X)
    roots = [complex(z) for z in sp.Poly(cub, X).nroots(n=30)]
    real = [z.real for z in roots if abs(z.imag) < 1e-20]
    mp.dps = 30
    Fv = lambda x: -x + mlog(1 + x * x) / 2 + mpf(1) / 400 * x * x + (1 if x != 0 else 0)
    cand = {xr: Fv(mpf(xr)) for xr in real}
    xs = max(real)
    rep(
        "S8h stationary points of F on x!=0 and global minimizer",
        len(real) >= 1 and min(cand.values()) < 0,
        f"(real roots of f': {[round(z, 6) for z in real]}, F(x*) = {float(cand[xs]):.6f} at x* = {xs:.6f}; F is coercive and F(0)=0)",
    )

    # R1 stationarity-radius formula against brute-force componentwise prox sets
    rng = np.random.default_rng(0)
    bad = 0
    ntr = 200000
    for _ in range(ntr):
        lam_ = rng.uniform(0.1, 2)
        gam = rng.uniform(0.05, 5)
        x = rng.normal(size=1)[0] * (rng.random() < 0.6)
        gr = rng.normal(size=1)[0] * (rng.random() < 0.7)
        if rng.random() < 0.2:  # force a tie
            if x != 0:
                gr = 0.0
                gam = x * x / (2 * lam_)
            elif gr != 0:
                gam = 2 * lam_ / gr**2
        y = x - gam * gr
        thr = np.sqrt(2 * gam * lam_)
        S = {y} if abs(y) > thr + 1e-12 else ({0.0} if abs(y) < thr - 1e-12 else {0.0, y})
        member = any(abs(v - x) < 1e-9 for v in S)
        single = len(S) == 1 and member
        rad = (x * x / (2 * lam_)) if x != 0 else (2 * lam_ / gr**2 if gr != 0 else np.inf)
        statJ = (x == 0) or (gr == 0)
        pred_member = statJ and gam <= rad * (1 + 1e-9)
        pred_single = statJ and gam < rad * (1 - 1e-9)
        if member != pred_member or (single != pred_single and abs(gam - rad) > 1e-9 * rad):
            bad += 1
    rep(
        "R1 stationarity-radius formula (componentwise, incl. ties)",
        bad == 0,
        f"({ntr} random cases, {bad} mismatches)",
    )

    # R2-R5 on random instances of the model
    cnt = dict(sd=0, step=0, chg=0, cntbound=0, runs=0, stat=0, statpts=0, ihtviol=0)
    for seed in range(30):
        rng = np.random.default_rng(seed)
        mm, nn = rng.integers(40, 120), rng.integers(60, 200)
        A = rng.normal(size=(mm, nn))
        xt = np.zeros(nn)
        xt[:5] = rng.normal(size=5) * 2
        b = np.sign(A @ xt + 0.5 * rng.normal(size=mm))
        b[b == 0] = 1
        P = L.ClassificationProblem(A, b, mu=1e-3)
        lam_ = rng.uniform(0.2, 1.0) * lam0(P)
        # R2 proximal-step bound pointwise bound ||d|| <= 2 g ||grad f(x)|| + sqrt(2 g (g(x) - g_inf)), g_inf = 0
        for _ in range(50):
            x = rng.normal(size=nn) * (rng.random(nn) < 0.2)
            gm = 10 ** rng.uniform(-3, 3)
            gx = P.grad_from_margins(P.margins(x), x)
            z = L.prox(x - gm * gx, gm, lam_, "l0")
            if (
                np.linalg.norm(z - x)
                > 2 * gm * np.linalg.norm(gx) + np.sqrt(2 * gm * lam_ * np.count_nonzero(x)) + 1e-9
            ):
                cnt["step"] += 1
        for name, kw in [
            ("LAPG", {}),
            ("LAPG-P", dict(probe=True)),
            ("NLAPG-P", dict(nm_p=0.2, probe=True)),
        ]:
            rr = L.lapg(P, lam_, "l0", trace=True, max_iter=3000, **kw)
            cnt["runs"] += 1
            Xs, gs = rr["X"], rr["gammas"]
            sig = 0.25
            pmin = kw.get("nm_p", 1.0)
            sw = rr.get("switch_iter", -1)
            Fv_ = [P.value(x) + lam_ * np.count_nonzero(x) for x in Xs]
            Ph = Fv_[0]
            changes = 0
            for k in range(len(Xs) - 1):
                dk = Xs[k + 1] - Xs[k]
                ref = Ph if pmin < 1 else Fv_[k]
                if Fv_[k + 1] > ref - sig / (2 * gs[k]) * (dk @ dk) + 1e-10 * max(1, abs(ref)):
                    cnt["sd"] += 1  # sufficient decrease
                if k >= 1 and not np.array_equal(Xs[k + 1] != 0, Xs[k] != 0):
                    changes += 1
                    n_enter = int(np.sum((Xs[k + 1] != 0) & (Xs[k] == 0)))
                    if ref - Fv_[k + 1] < sig * lam_ * n_enter - 1e-10:
                        cnt["chg"] += 1  # support-entry decrease: decrease >= sigma*lam*|E_k|
                pk = pmin if (pmin < 1 and (sw < 0 or k <= sw)) else 1.0
                Ph = (1 - pk) * Ph + pk * Fv_[k + 1]
            if len(gs) > 1:
                # support-change bound: #changes <= 2 (Phi_1 - F_inf)/(p_min sigma lam) + |supp(x^1)|, with F_inf >= 0
                Phi1 = Fv_[0] + (pmin if pmin < 1 else 1.0) * (Fv_[1] - Fv_[0])
                bound = 2 * Phi1 / (pmin * sig * lam_) + np.count_nonzero(Xs[1])
                if changes > bound:
                    cnt["cntbound"] += 1
            # R5 finite-grid descent certificate at the final point: regular Gamma-descent stationarity on the probing grid
            if kw.get("probe"):
                x = Xs[-1]
                gx = P.grad_from_margins(P.margins(x), x)
                Fx = Fv_[-1]
                Gam = 1e6 * 4.0 ** (-np.arange(20.0))
                for gm in Gam:
                    y = x - gm * gx
                    thr = np.sqrt(2 * gm * lam_)
                    if np.min(np.abs(np.abs(y) - thr)) < 1e-12 * thr:
                        continue  # tie: not in Gamma_reg
                    u = L.prox(y, gm, lam_, "l0")
                    cnt["statpts"] += 1
                    if (
                        P.value(u) + lam_ * np.count_nonzero(u)
                        < Fx - sig / (2 * gm) * ((u - x) @ (u - x)) - 1e-10
                    ):
                        cnt["stat"] += 1
        ri = L.pg_const(P, lam_, "l0", max_iter=5000)
        x = ri["x"]
        gx = P.grad_from_margins(P.margins(x), x)
        Fx = P.value(x) + lam_ * np.count_nonzero(x)
        cnt["ihtviol"] += any(
            P.value(u) + lam_ * np.count_nonzero(u) < Fx - 0.25 / (2 * gm) * ((u - x) @ (u - x))
            for gm in 1e6 * 4.0 ** (-np.arange(20.0))
            for u in [L.prox(x - gm * gx, gm, lam_, "l0")]
        )
    rep(
        "R2 proximal-step bound trial-point bound",
        cnt["step"] == 0,
        f"(1500 random points/step sizes, {cnt['step']} violations)",
    )
    rep(
        "R3 sufficient-decrease inequality sufficient decrease at every step",
        cnt["sd"] == 0,
        f"({cnt['runs']} runs, {cnt['sd']} violations)",
    )
    rep(
        "R4 support-change bound decrease at every entry and total count of support changes",
        cnt["chg"] == 0 and cnt["cntbound"] == 0,
        f"({cnt['chg']} per-change violations, {cnt['cntbound']} count violations)",
    )
    rep(
        "R5 finite-grid descent certificate at final points of LAPG-P / NLAPG-P",
        cnt["stat"] == 0,
        f"({cnt['statpts']} (point, step size) pairs, {cnt['stat']} violations)",
    )
    out.append(
        f"[INFO] IHT final points that violate (Gamma,1/4)-descent stationarity: {cnt['ihtviol']} of 30"
    )
    print(out[-1])
    out.append("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
    (Path(RES) / "verification.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(out[-1])
    if not ok_all:
        raise SystemExit(1)


def main(argv=None):
    """Parse the command line and execute one reproducible experiment group."""
    global RES
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("example").set_defaults(fn=example)
    sub.add_parser("verify").set_defaults(fn=verify)
    sub.add_parser("ablation").set_defaults(fn=ablation)
    sub.add_parser("profile").set_defaults(fn=profile)
    p = sub.add_parser("synth")
    p.add_argument("--m", type=int, default=500)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seeds", type=index_range, default="0:10")
    p.set_defaults(fn=synth)
    p = sub.add_parser("real")
    p.add_argument("--data", choices=("breast_cancer", "digits_parity"), default="breast_cancer")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--reps", type=index_range, default="0:20")
    p.set_defaults(fn=real)
    for parser in sub.choices.values():
        parser.add_argument(
            "--save-inputs",
            action="store_true",
            help="Archive inputs and final solution vectors for synthetic/real experiments.",
        )
        parser.add_argument(
            "--results-dir",
            default=os.environ.get("LAPG_RESULTS_DIR", RES),
            help="Output directory (default: results next to this script).",
        )
    a = ap.parse_args(argv)
    if hasattr(a, "noise") and not 0 <= a.noise <= 1:
        ap.error("--noise must lie in [0, 1].")
    RES = str(Path(a.results_dir).expanduser().resolve())
    os.makedirs(RES, exist_ok=True)
    # Metadata filenames distinguish command groups and chunks.
    metadata_name = (
        "metadata_"
        + "_".join(
            str(value).replace(":", "-")
            for key, value in vars(a).items()
            if key not in ("fn", "results_dir")
        )
        + ".json"
    )
    _write_json(_run_metadata(a), metadata_name)
    a.fn(a)


if __name__ == "__main__":
    main()
