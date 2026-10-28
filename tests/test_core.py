"""Tests for numerical behavior and the experimental protocol."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import experiments as E
import lapg as L
import report


class OneDimensionalExample:
    """Objective from the manuscript's one-dimensional counterexample."""

    n = 1
    L = 1.005

    def reset_counts(self):
        self.counts = {"feval": 0, "geval": 0}

    def margins(self, x):
        return x.copy()

    def value_from_margins(self, t, x):
        self.counts["feval"] += 1
        return float(-x[0] + 0.5 * np.log1p(x[0] ** 2) + 0.0025 * x[0] ** 2)

    def value(self, x):
        return self.value_from_margins(x, x)

    def grad_from_margins(self, t, x):
        self.counts["geval"] += 1
        return -1 + x / (1 + x * x) + 0.005 * x


class CoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.A = rng.normal(size=(12, 5))
        self.b = rng.choice([-1.0, 1.0], size=12)

    def test_sigmoid_gradient(self):
        problem = L.ClassificationProblem(self.A, self.b)
        self.check_gradient(problem)

    def test_logistic_gradient(self):
        problem = L.LogisticProblem(self.A, self.b)
        self.check_gradient(problem)

    def check_gradient(self, problem):
        x = np.linspace(-0.2, 0.4, problem.n)
        epsilon = 1e-6
        numerical = []
        for direction in np.eye(problem.n):
            numerical.append(
                (problem.value(x + epsilon * direction) - problem.value(x - epsilon * direction))
                / (2 * epsilon)
            )
        gradient = problem.grad_from_margins(problem.margins(x), x)
        np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=1e-6)

    def test_hard_threshold_ties_are_zero(self):
        np.testing.assert_array_equal(
            L.prox_l0(np.array([-2.0, -1.0, 1.0, 2.0]), 0.5, 1), [-2, 0, 0, 2]
        )

    def test_l1_soft_threshold(self):
        np.testing.assert_array_equal(
            L.prox(np.array([-2.0, -1.0, 0.0, 2.0]), 1, 1, "l1"), [-1, 0, 0, 1]
        )

    def test_manuscript_grid(self):
        grid = L.probe_grid(1e-6, 1e6, 4.0)
        np.testing.assert_allclose(grid, 1e6 * 4.0 ** (-np.arange(20)))

    def test_final_probe_accepts_an_improvement(self):
        # A deliberately loose residual tolerance forces the terminal-probe
        # path immediately after the first accepted step. An improving probe
        # must still be accepted on this path.
        problem = OneDimensionalExample()
        first = L.lapg(problem, 1, "l0", probe=True, gamma_hi=16, tol=1e3, max_iter=1)
        second = L.lapg(problem, 1, "l0", probe=True, gamma_hi=16, tol=1e3, max_iter=2)
        self.assertEqual(second["n_final_probes"], 1)
        self.assertEqual(second["n_probe_moves"], 2)
        self.assertLess(second["hist"][-1, 1], first["hist"][-1, 1])
        self.assertFalse(second["converged"])

    def test_final_probe_is_in_evaluation_counts(self):
        problem = OneDimensionalExample()
        result = L.lapg(problem, 1, "l0", probe=True, gamma_hi=16, max_iter=1000)
        self.assertTrue(result["converged"])
        self.assertEqual(int(result["hist"][-1, 4]), problem.counts["feval"])
        self.assertEqual(int(result["hist"][-1, 5]), problem.counts["geval"])

    def test_monotone_decrease_and_certificate(self):
        problem, _ = E.synth_problem(4, 25, 30)
        penalty = 0.8 * E.lam0(problem)
        result = L.lapg(problem, penalty, "l0", probe=True, max_iter=5000)
        self.assertTrue(result["converged"])
        self.assertTrue(np.all(np.diff(result["hist"][:, 1]) <= 1e-12))
        self.assertTrue(L.descent_check(problem, result["x"], penalty)[0])

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            L.ClassificationProblem(self.A, np.zeros(12))
        with self.assertRaises(ValueError):
            L.probe_grid(1, 10, 1)
        with self.assertRaises(ValueError):
            E.synthetic_arrays(0, 10, 19)
        with self.assertRaises(ValueError):
            E.real_arrays("typo", 0, 0)

    def test_six_methods_and_reference(self):
        problem, _ = E.synth_problem(0, 20, 25)
        rows = E.run_instance(problem, 0.8 * E.lam0(problem))
        self.assertEqual([row["method"] for row in rows], E.METHODS)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["Fref"] == min(item["F"] for item in rows) for row in rows))

    def test_synthetic_generation_repeatable(self):
        for first, second in zip(E.synthetic_arrays(7, 20, 25), E.synthetic_arrays(7, 20, 25)):
            np.testing.assert_array_equal(first, second)

    def test_real_split_and_training_only_standardization(self):
        clean = E.real_arrays("breast_cancer", 0, 0.0)
        noisy = E.real_arrays("breast_cancer", 0, 0.2)
        self.assertFalse(set(clean["train_indices"]) & set(clean["test_indices"]))
        self.assertEqual(len(clean["train_indices"]) + len(clean["test_indices"]), 569)
        np.testing.assert_allclose(clean["X_train"].mean(0), 0, atol=1e-10)
        np.testing.assert_array_equal(clean["y_test"], noisy["y_test"])
        np.testing.assert_array_equal(
            clean["y_train"] != noisy["y_train"], noisy["flipped_train_mask"]
        )

    def test_input_archive_contains_no_pickle_arrays(self):
        original = E.RES
        with tempfile.TemporaryDirectory() as directory:
            try:
                E.RES = directory
                E._save_arrays("case.npz", x=np.arange(4), penalty=np.asarray(0.1))
                with np.load(Path(directory) / "inputs" / "case.npz", allow_pickle=False) as data:
                    np.testing.assert_array_equal(data["x"], np.arange(4))
            finally:
                E.RES = original

    def test_overlapping_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in ("a", "b"):
                pd.DataFrame([dict(seed=0, method="IHT")]).to_csv(
                    root / f"synth_{suffix}.csv", index=False
                )
            with self.assertRaises(ValueError):
                report.read_csv_group(root, "synth_*.csv", ["seed", "method"])

    def test_reference_cannot_include_extra_restarts(self):
        data = pd.DataFrame(
            [
                dict(method=method, F=0.2 + index / 100, Fref=0.1, time=1.0, conv=True, seed=0)
                for index, method in enumerate(E.METHODS)
            ]
        )
        with self.assertRaises(ValueError):
            report.validate_data(data, ["seed"], complete=True)


if __name__ == "__main__":
    unittest.main()
