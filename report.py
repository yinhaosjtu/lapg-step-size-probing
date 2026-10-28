"""Generate the manuscript tables, figures, and paired comparisons.

Inputs are the CSV/JSON files created by experiments.py. No manuscript source
is rewritten. Outputs are table-body fragments, figures, and a JSON summary.
By default all experiment groups are required. Use --allow-partial to
generate a report from the available experiment groups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments import ABL, METHODS

COLORS = dict(zip(METHODS, ["#7F7F7F", "#EE6677", "#66CCEE", "#4477AA", "#CCBB44", "#228833"]))
STYLES = dict(zip(METHODS, [":", "-.", "--", "-", (0, (5, 1.5)), "-"]))
REAL_NAMES = {"breast_cancer": "Breast cancer", "digits_parity": "Digits parity"}
ROW_END = " \\\\"


def read_csv_group(directory: Path, pattern: str, keys: list[str]) -> pd.DataFrame | None:
    """Load chunks and reject overlapping instance/method records."""
    files = sorted(directory.glob(pattern))
    if not files:
        return None
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    missing = set(keys) - set(data.columns)
    if missing:
        raise ValueError(f"{pattern}: missing columns {sorted(missing)}")
    if data.duplicated(keys).any():
        raise ValueError(f"{pattern}: overlapping result chunks; use a fresh results directory.")
    return data


def validate_data(data: pd.DataFrame, keys: list[str], complete: bool) -> None:
    """Check method coverage, finite metrics, and the six-method denominator."""
    for identifier, group in data.groupby(keys):
        if set(group.method) != set(METHODS) or len(group) != 6:
            raise ValueError(
                f"Instance {identifier} does not contain exactly the six manuscript methods."
            )
        if not np.all(np.isfinite(group[["F", "Fref", "time"]].to_numpy())):
            raise ValueError(f"Instance {identifier} contains nonfinite primary results.")
        if not np.allclose(group.Fref, group.F.min(), rtol=1e-12, atol=1e-14):
            raise ValueError(f"Instance {identifier}: Fref must equal min(F) of the six methods.")
        if group.Fref.min() <= 0:
            raise ValueError("The relative-gap definition requires positive Fref.")
        if complete and not bool(group.loc[group.method != "IHT", "conv"].all()):
            raise ValueError(
                f"Instance {identifier}: an adaptive method hit a safeguard; inspect its run."
            )
    data["gap"] = 100.0 * (data.F - data.Fref) / data.Fref


def _write_table(directory: Path, name: str, lines: list[str]) -> None:
    (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def synthetic_table(data: pd.DataFrame, directory: Path) -> pd.DataFrame:
    """Match the synthetic table: means, with median iterations and times."""
    summary = data.groupby(["m", "n", "method"], sort=True).agg(
        F=("F", "mean"),
        gap=("gap", "mean"),
        nnz=("nnz", "mean"),
        tp=("tp", "mean"),
        iters=("iters", "median"),
        time=("time", "median"),
        radiusL=("radiusL", "mean"),
    )
    lines = []
    for (m, n), group in summary.groupby(level=["m", "n"]):
        group = group.droplevel(["m", "n"])
        for index, method in enumerate(METHODS):
            row = group.loc[method]
            label = f"\\multirow{{6}}{{*}}{{${m}\\times{n}$}}" if index == 0 else ""
            lines.append(
                f"{label} & {method} & {row.F:.4f} & {row.gap:.1f} & {row.nnz:.1f} & {row.tp:.1f} & {row.iters:.0f} & {row.time:.2f} & {row.radiusL:.1f}"
                + ROW_END
            )
        lines.append("\\midrule")
    _write_table(directory, "table_synth.tex", lines[:-1])
    summary.to_csv(directory / "summary_synthetic.csv")
    return summary


def real_tables(data: pd.DataFrame, directory: Path) -> None:
    """Write the gap and accuracy table bodies, including logistic-l1."""
    gap_lines, accuracy_lines = [], []
    for (dataset, noise, relative_lambda), group in data.groupby(["data", "noise", "lam_rel"]):
        summary = group.groupby("method").mean(numeric_only=True)
        prefix = f"{REAL_NAMES[dataset]} & {noise:.1f} & {relative_lambda:.2f}"
        best = summary.gap.min()
        cells = [
            (
                f"\\textbf{{{summary.loc[method, 'gap']:.2f}}}"
                if np.isclose(summary.loc[method, "gap"], best, atol=1e-9, rtol=0)
                else f"{summary.loc[method, 'gap']:.2f}"
            )
            for method in METHODS
        ]
        gap_lines.append(prefix + " & " + " & ".join(cells) + ROW_END)
        cells = [
            f"{100*summary.loc[method, 'acc']:.1f} ({summary.loc[method, 'nnz']:.0f})"
            for method in ["IHT", "SpaRSA", "LAPG", "NLAPG-P"]
        ]
        # One identical convex-reference result is stored with each of the six methods.
        reference = group.drop_duplicates("rep")
        cells.append(
            f"{100*reference.logistic_acc.mean():.1f} ({reference.logistic_nnz.mean():.0f})"
        )
        accuracy_lines.append(prefix + " & " + " & ".join(cells) + ROW_END)
    _write_table(directory, "table_real_gap.tex", gap_lines)
    _write_table(directory, "table_real_acc.tex", accuracy_lines)
    data.groupby(["data", "noise", "lam_rel", "method"]).mean(numeric_only=True).to_csv(
        directory / "summary_real.csv"
    )


def ablation_table(data: pd.DataFrame, directory: Path) -> None:
    """Use the best of eight variants on each of the 30 sensitivity instances."""
    data = data.copy()
    data["best"] = data.groupby(["group", "inst"]).F.transform("min")
    data["gap"] = 100 * (data.F - data.best) / data.best
    per_group = data.groupby(["variant", "group"]).gap.mean().unstack()
    overall = data.groupby("variant").agg(
        gap=("gap", "mean"), max_gap=("gap", "max"), feval=("feval", "median")
    )
    labels = [
        "reference",
        "$T=1$",
        "$T=50$",
        "grid ratio $2$",
        "grid ratio $16$",
        "$p_{\\min}=0.5$",
        "$p_{\\min}=0.9$",
        "no probing (NLAPG)",
    ]
    lines = []
    for name, label in zip(ABL, labels):
        group, row = per_group.loc[name], overall.loc[name]
        values = [
            group.get(key, np.nan)
            for key in ["synthetic", "breast cancer p=0.0", "breast cancer p=0.2"]
        ]
        lines.append(
            label
            + " & "
            + " & ".join(f"{value:.2f}" for value in values + [row.gap, row.max_gap])
            + f" & {row.feval:.0f}"
            + ROW_END
        )
    _write_table(directory, "table_ablation.tex", lines)
    overall.to_csv(directory / "summary_ablation.csv")


def paired_tests(data: pd.DataFrame, directory: Path) -> None:
    """Run paired, two-sided Wilcoxon tests on accuracy across splits."""
    rows = []
    for setting, group in data.groupby(["data", "noise", "lam_rel"]):
        pivot = group.pivot(index="rep", columns="method", values="acc")
        reference = group.drop_duplicates("rep").set_index("rep").logistic_acc
        for comparator in ["SpaRSA", "IHT", "logistic-l1"]:
            a = pivot["NLAPG-P"]
            b = reference.reindex(a.index) if comparator == "logistic-l1" else pivot[comparator]
            differences = np.asarray(a - b)
            probability = (
                1.0
                if np.all(differences == 0)
                else float(wilcoxon(a, b, zero_method="zsplit", alternative="two-sided").pvalue)
            )
            rows.append(
                dict(
                    data=setting[0],
                    noise=setting[1],
                    lam_rel=setting[2],
                    comparator=comparator,
                    n=len(a),
                    mean_accuracy_difference=float(differences.mean()),
                    p_value=probability,
                    bonferroni_eight=min(1.0, 8 * probability),
                )
            )
    pd.DataFrame(rows).to_csv(directory / "wilcoxon_accuracy.csv", index=False)


def certification_summary(synthetic, real, directory):
    """Select exactly the manuscript's 30 synthetic and 40 breast-cancer cases."""
    parts = {}
    if synthetic is not None:
        parts["synthetic"] = synthetic[
            (synthetic.m == 500) & (synthetic.n == 2000) & (synthetic.seed < 30)
        ]
    if real is not None:
        parts["breast_cancer"] = real[(real.data == "breast_cancer") & (real.rep < 10)]
    rows = []
    for title, data in parts.items():
        for method in METHODS:
            subset = data[data.method == method]
            if not subset.empty:
                rows.append(
                    dict(
                        dataset=title,
                        method=method,
                        instances=len(subset),
                        certified=int(subset.certified.sum()),
                        median_relative_decrease=float(subset.max_rel_decrease.median()),
                        maximum_relative_decrease=float(subset.max_rel_decrease.max()),
                    )
                )
    pd.DataFrame(rows).to_csv(directory / "certification.csv", index=False)


def configure_plots():
    """Use an embedded-font, color-distinguishable style without external TeX."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def example_figure(source: Path, output: Path):
    """Plot the one-dimensional illustration using computed example results."""
    data = json.loads(source.read_text(encoding="utf-8"))
    x = np.linspace(-20, 260, 3000)
    objective = -x + 0.5 * np.log1p(x * x) + 0.0025 * x * x + 1
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.plot(x, objective, label=r"$F(x)$, $x\ne0$")
    axis.plot([0], [0], "o", label=r"$\bar{x}=0$: IHT and LAPG")
    steps = [4, 16, 64, 256]
    axis.plot(
        [data["probes"][str(step)][0] for step in steps],
        [data["probes"][str(step)][1] for step in steps],
        "s",
        label="Probing points",
    )
    axis.plot([data["x_global"]], [data["F_global"]], "*", markersize=12, label="Global minimizer")
    axis.set(xlabel=r"$x$", ylabel=r"$F(x)$", ylim=(-110, 25))
    axis.legend()
    figure.savefig(output)
    plt.close(figure)


def performance_figure(parts, output):
    """Plot the two manuscript panels using the six-method reference value."""
    figure, axes = plt.subplots(
        1, len(parts), figsize=(5.3 * len(parts), 3.8), squeeze=False, sharey=True
    )
    for axis, (title, data) in zip(axes[0], parts):
        thresholds = np.linspace(1, 1.10, 400)
        for method in METHODS:
            subset = data[data.method == method]
            ratio = np.asarray(subset.F / subset.Fref)
            axis.plot(
                thresholds,
                [(ratio <= threshold + 1e-12).mean() for threshold in thresholds],
                label=method,
                color=COLORS[method],
                ls=STYLES[method],
            )
        axis.set(title=title, xlabel=r"$\tau$", xlim=(1, 1.10), ylim=(0, 1.02))
    axes[0, 0].set_ylabel(r"Fraction with $F\leq\tau F_{\mathrm{best}}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    figure.savefig(output)
    plt.close(figure)


def descent_figure(source: Path, output: Path):
    """Draw curves and method-specific markers at the actual probing grid."""
    data = json.loads(source.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, (title, results) in zip(axes, data.items()):
        axis.axhspan(0, 3, color="#EE6677", alpha=0.08)
        for method in METHODS:
            result = results[method]
            axis.plot(
                result["gammaL"],
                result["excess"],
                label=method,
                color=COLORS[method],
                ls=STYLES[method],
            )
            axis.plot(
                result["grid_gammaL"], result["grid_excess"], "o", ms=3, color=COLORS[method]
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(
            xscale="log",
            yscale="symlog",
            ylim=(-1, 3),
            xlim=(0.1, 1e6),
            title=title,
            xlabel=r"$\gamma L_f$",
        )
    for axis in axes:
        axis.set_yscale("symlog", linthresh=1e-3)
    axes[0].set_ylabel("Relative excess decrease")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    figure.savefig(output)
    plt.close(figure)


def main(argv=None):
    """Validate stored experiments and generate outputs without modifying a paper."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--output-dir", type=Path, default=root / "reports")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Generate outputs from the available experiment groups.",
    )
    args = parser.parse_args(argv)
    synthetic = read_csv_group(args.results_dir, "synth_*.csv", ["m", "n", "seed", "method"])
    real = read_csv_group(
        args.results_dir, "real_*.csv", ["data", "noise", "rep", "lam_rel", "method"]
    )
    ablation = read_csv_group(args.results_dir, "ablation.csv", ["group", "inst", "variant"])
    if not args.allow_partial and any(data is None for data in (synthetic, real, ablation)):
        parser.error(
            "The complete report requires synthetic, real-data, and ablation results. Run run_all.sh or use --allow-partial for development."
        )
    if synthetic is not None:
        validate_data(synthetic, ["m", "n", "seed"], not args.allow_partial)
        if not args.allow_partial:
            expected = {(500, 2000, seed) for seed in range(30)} | {
                (1000, 5000, seed) for seed in range(20)
            }
            observed = set(map(tuple, synthetic[["m", "n", "seed"]].drop_duplicates().to_numpy()))
            if observed != expected:
                parser.error(
                    "Synthetic coverage must be 30 cases at 500x2000 and 20 at 1000x5000."
                )
    if real is not None:
        validate_data(real, ["data", "noise", "rep", "lam_rel"], not args.allow_partial)
        if not args.allow_partial:
            expected = {
                (dataset, noise, rep, penalty)
                for dataset in REAL_NAMES
                for noise in (0.0, 0.2)
                for rep in range(20)
                for penalty in (0.25, 0.5)
            }
            observed = set(
                map(tuple, real[["data", "noise", "rep", "lam_rel"]].drop_duplicates().to_numpy())
            )
            if observed != expected or not bool(real.logistic_converged.all()):
                parser.error("Real-data coverage or logistic-reference convergence is incomplete.")
    if ablation is not None and not args.allow_partial:
        expected = {
            (group, f"{prefix}{index}", variant)
            for group, prefix in [
                ("synthetic", "s"),
                ("breast cancer p=0.0", "r"),
                ("breast cancer p=0.2", "r"),
            ]
            for index in range(10)
            for variant in ABL
        }
        observed = set(map(tuple, ablation[["group", "inst", "variant"]].to_numpy()))
        if observed != expected or not bool(ablation.conv.all()):
            parser.error("The ablation requires eight converged variants on exactly 30 instances.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    configure_plots()
    if synthetic is not None:
        synthetic_table(synthetic, args.output_dir)
    if real is not None:
        real_tables(real, args.output_dir)
        paired_tests(real, args.output_dir)
    if ablation is not None:
        ablation_table(ablation, args.output_dir)
    certification_summary(synthetic, real, args.output_dir)
    parts = [
        (title, data)
        for title, data in [("Synthetic data", synthetic), ("Real data", real)]
        if data is not None
    ]
    if parts:
        performance_figure(parts, figures / "fig_profiles.pdf")
    for filename, function, output in [
        ("example1d.json", example_figure, "fig_example.pdf"),
        ("profile.json", descent_figure, "fig_profile.pdf"),
    ]:
        source = args.results_dir / filename
        if source.exists():
            function(source, figures / output)
        elif not args.allow_partial:
            parser.error(f"Missing {filename}; run the corresponding experiments command.")
    counts = {
        "partial": args.allow_partial,
        "synthetic_rows": 0 if synthetic is None else len(synthetic),
        "real_rows": 0 if real is None else len(real),
        "ablation_rows": 0 if ablation is None else len(ablation),
    }
    (args.output_dir / "report_summary.json").write_text(
        json.dumps(counts, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
