# Proximal gradient methods with step-size probing

Numerical algorithms and experiments for **“Proximal gradient methods with
step-size probing for ℓ₀-regularized problems”**, by Hao Yin and Zishang Li.

The repository includes proximal gradient solvers, synthetic and real-data
experiments, numerical verification routines, and tools for producing tables
and figures. Experiment inputs, final solution vectors, and environment
metadata can be saved alongside the numerical results.

## Optimization model

The principal experiments solve

\[
\min_x\; F(x)=\frac1m\sum_{i=1}^{m}\frac{1}{1+\exp(b_i a_i^\top x)}
+\frac{10^{-3}}2\|x\|^2+\lambda\|x\|_0,
\qquad b_i\in\{-1,+1\}.
\]

The ℓ₀ proximal map is implemented by hard thresholding, with equality at the
threshold resolved to zero.

| Method | Step-size and acceptance rule | Probing |
| --- | --- | --- |
| IHT | Constant step `1 / L_f` | No |
| SpaRSA | Absolute-curvature BB step; maximum of five recent objective values | No |
| LAPG | BB trial step; monotone smooth-model backtracking | No |
| LAPG-P | Same as LAPG | Yes |
| NLAPG | Averaged nonmonotone acceptance, followed by monotone acceptance | No |
| NLAPG-P | Same as NLAPG | Yes |

A logistic–ℓ₁ model provides a separate test-accuracy reference. Its objective
values are excluded from the six-method objective comparison.

## Installation

Use **Python 3.11 or newer** and install the pinned dependencies in a virtual
environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. The Python
commands work directly; the batch script requires Bash, available through WSL.

For single-thread numerical execution:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

The real datasets are bundled with scikit-learn. Figure generation uses
Matplotlib and does not require an external LaTeX installation.

## Quick start

Run commands from the repository root:

```bash
python experiments.py example --results-dir results-demo
python experiments.py synth --m 30 --n 40 --seeds 0:1 \
  --save-inputs --results-dir results-demo
python report.py --results-dir results-demo --output-dir reports-demo --allow-partial
```

The `30 × 40` case is a small example. Index ranges are half-open: `0:20`
selects indices 0 through 19.

## Experiment suite

```bash
bash run_all.sh
```

The batch script executes the following experiments, saves their inputs and
results, and generates the report:

| Experiment | Configuration |
| --- | --- |
| Synthetic, small | `(m, n) = (500, 2000)`; 30 instances; `lambda = 0.8 lambda_0` |
| Synthetic, large | `(m, n) = (1000, 5000)`; 20 instances; `lambda = 0.8 lambda_0` |
| Real data | Two datasets; 20 splits each; noise probabilities `0` and `0.2`; `lambda/lambda_0` equal to `0.25` and `0.5` |
| Descent-stationarity summaries | 30 small synthetic instances and 40 breast-cancer instances from splits `0:10` |
| Parameter sensitivity | Eight NLAPG-P variants on ten synthetic and twenty breast-cancer instances |
| Descent-profile figure | One synthetic instance and a breast-cancer instance where LAPG fails the finite-grid stationarity check |
| Illustration | The one-dimensional example |

There are 50 synthetic and 160 real-data optimization instances, each evaluated
by six methods. Sensitivity experiments are evaluated separately.

The wrapper defaults to `JOBS=1`, sets the BLAS thread counts to one, writes
per-command logs, and resumes completed chunks using `.done` markers.

```bash
LAPG_RESULTS_DIR=results-release PYTHON=python bash run_all.sh
```

For parallel execution, set `JOBS` to the desired number of processes. Use
`JOBS=1` for single-process timing measurements. Synthetic design matrices in
the full suite occupy approximately 1.04 GB before compression.

Individual experiment commands are also available:

```bash
python experiments.py synth --m 500 --n 2000 --seeds 0:30 --save-inputs
python experiments.py synth --m 1000 --n 5000 --seeds 0:20 --save-inputs
python experiments.py real --data breast_cancer --noise 0.0 --reps 0:20 --save-inputs
python experiments.py real --data breast_cancer --noise 0.2 --reps 0:20 --save-inputs
python experiments.py real --data digits_parity --noise 0.0 --reps 0:20 --save-inputs
python experiments.py real --data digits_parity --noise 0.2 --reps 0:20 --save-inputs
python experiments.py ablation --save-inputs
python experiments.py profile --save-inputs
python experiments.py example
python report.py
```

Use separate result directories for chunked and unchunked executions to avoid
overlapping records. The reporter checks instance coverage and rejects
duplicate instance/method rows.

## Algorithm parameters

Default parameters are `rho = eta = 0.5`, `sigma = 0.25`, trial-step bounds
`[1e-6, 1e6]`, probing period `T = 10`, and
`Gamma = {1e6 * 4**(-j): j = 0, ..., 19}`. Nonmonotone methods use averaging
parameter `0.2`, switching to `1` after 20 consecutive support-preserving
steps. SpaRSA uses memory 5 and decrease parameter `1e-2`.

All principal runs start at zero. The stopping tolerance is `1e-6` for the
external proximal-gradient residual, or, for adaptive methods, the accepted
step residual. Probing methods perform an additional probing step before
successful termination. Real-data IHT has a 5-second algorithm-time budget.
Iteration caps are recorded through the convergence flag.

The logistic reference minimizes averaged logistic loss plus ℓ₁ with
`lambda_1 = 0.05 * ||grad f_log(0)||_infinity`, without an intercept or a ridge
term, using the same preprocessed data and an adaptive proximal solver.

## Datasets and preprocessing

| Dataset | Repository and identifier | Loader |
| --- | --- | --- |
| Breast Cancer Wisconsin (Diagnostic) | [UCI Machine Learning Repository; DOI 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) | `sklearn.datasets.load_breast_cancer`; 569 samples and 30 original features |
| Optical Recognition of Handwritten Digits | [UCI Machine Learning Repository; DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49) | `sklearn.datasets.load_digits`; the 1,797-sample test-set subset with 64 original features |

Dataset documentation:
[breast cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
and [digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

Degree-two polynomial features include original features, squares, and
pairwise products, without a bias column. This gives 495 and 2,144 columns
before constant-column removal. Data are split into stratified 70% training
and 30% test partitions. Constant-column selection and standardization are
fitted on the training partition only. Label noise affects training labels;
test labels remain unchanged. Positive labels indicate benign breast-cancer
samples and even digits. Zero-score predictions receive the positive label.

Synthetic instances use generator seed `1000 + seed`, a planted vector with
20 nonzero entries of random signs and magnitudes in `[1, 2)`, and Gaussian
label noise with standard deviation 0.5. Real-data splits use
`random_state=rep`; label flips use generator seed `500 + rep`.

See [data/README.md](data/README.md) for archive schemas and data provenance.
Third-party datasets and packages retain their respective licenses.

## Outputs

| Output | Contents |
| --- | --- |
| `results/synth_*.csv` | Synthetic results, one row per instance and method |
| `results/real_*.csv` | Real-data results and logistic-reference accuracy columns |
| `results/ablation.csv` | Sensitivity results, one row per instance and variant |
| `results/example1d.json`, `results/profile.json` | Numerical inputs to the illustration and descent-profile figures |
| `results/metadata_*.json` | Command, platform, dependency versions, thread settings, and source SHA-256 hashes |
| `results/inputs/*.npz` | Optional input arrays and final solution vectors |
| `reports/table_*.tex` | LaTeX table-body fragments |
| `reports/figures/*.pdf` | Generated figures |
| `reports/certification.csv` | Finite-grid descent-stationarity summaries |
| `reports/wilcoxon_accuracy.csv` | Paired accuracy comparisons |
| `reports/summary_*.csv` | Aggregated experiment results |

The CSV column `Fref` denotes `F_best`, the smallest final objective value
among the six compared methods on the same instance. Relative gaps are
`100 * (F - Fref) / Fref`. Sensitivity gaps instead use the best of the eight
variants on each instance.

`conv` records successful solver termination; `certified` records the separate
finite-grid descent-stationarity check with floating-point tolerances. The
stationarity-radius diagnostic additionally requires a vanishing gradient on
the nonzero support for its fixed-point interpretation.

Algorithm timings include terminal probing and exclude diagnostic history
logging, dataset preparation, and initial Lipschitz-bound computation. Time
limits are checked between iterations. Tables are exported as body fragments
for inclusion in the corresponding LaTeX table environments.

## Tests and numerical checks

```bash
python -m unittest discover -s tests -v
python experiments.py verify --results-dir results-checks
```

The verification command evaluates symbolic identities, interval calculations,
and randomized properties of the proximal maps and solvers.

Development tools are available through `requirements-dev.txt`:

```bash
python -m pip install -r requirements-dev.txt
python -m black --check .
python -m ruff check .
bash -n run_all.sh
```

## Citation

Author and software metadata are provided in [CITATION.cff](CITATION.cff).
Please cite the associated paper and the archived software release when using
this implementation.
