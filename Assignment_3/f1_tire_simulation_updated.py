#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from statsmodels.regression.linear_model import OLS

SEED = 12345
GAMMA_TRUE = 4.0
SCALE_0 = 30.0
Q_20 = 26.58  # Preserve the submitted code's rounded value.
BETA = np.array([0.6, -0.4, 0.8])
BETA_NAMES = ["beta_z", "beta_interaction", "beta_dirty"]
MODELS = ["correct_weibull", "omitted_dirty_weibull", "exponential", "naive_ols"]
PARAM_TRUTH = dict(zip(BETA_NAMES, BETA)) | {"gamma": GAMMA_TRUE}
COVARIANCES = ["model_based"]
SOURCE_COMMIT = "851d7563210739cba23bafcbff2a97d0e81e2f57"


def profiles():
    rows = []
    for temp in (25, 35, 45):
        for compound in (0, 1):
            z = (temp - Q_20) / 10
            x = np.array([z, compound * z, 0.5])
            truth = SCALE_0 * np.exp(-x @ BETA / GAMMA_TRUE
                                    + gammaln(1 + 1 / GAMMA_TRUE))
            rows.append((f"mean_T_t{temp}_c{compound}_d0.5", x, truth))
    return rows


def covariates(rng, n_races=20, stints_per_race=60):
    race = np.repeat(np.arange(n_races), stints_per_race)
    temperature = np.repeat(rng.normal(35, 10, n_races), stints_per_race)
    z = (temperature - Q_20) / 10
    compound = rng.binomial(1, 0.8, len(race))
    dirty = rng.beta(0.5, 0.4, len(race))
    return np.column_stack((z, compound * z, dirty)), race


def failure_times(rng, x):
    u = np.maximum(rng.uniform(size=len(x)), np.finfo(float).tiny)
    return SCALE_0 * np.exp(-x @ BETA / GAMMA_TRUE) * (-np.log(u)) ** (1 / GAMMA_TRUE)


def likelihood(par, y, delta, a, estimate_gamma, derivatives=False):
    """Exact NLL, gradient, and optional observed information.

    Time centering reduces intercept/shape numerical correlation. Returning
    infinity for overflow rejects an unsafe trial, without clipping the model.
    """
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        gamma = np.exp(par[-1]) if estimate_gamma else 1.0
        w = par[:-1] if estimate_gamma else par
        logt = np.log(y / SCALE_0)
        v = gamma * logt
        h = np.exp(a @ w + v)
        nll = np.sum(h - delta * (np.log(gamma) + a @ w + v - np.log(y)))
        gradient = a.T @ (h - delta)
        if estimate_gamma:
            gradient = np.r_[gradient, np.sum((h - delta) * v - delta)]
    if not np.isfinite(nll) or not np.isfinite(gradient).all():
        return (np.inf, np.zeros_like(par)) if not derivatives else None
    if not derivatives:
        return nll, gradient
    information = a.T @ (h[:, None] * a)
    if estimate_gamma:
        cross = a.T @ (h * v)
        bottom = np.sum(h * v * v + (h - delta) * v)
        information = np.block([[information, cross[:, None]],
                                [cross[None, :], np.array([[bottom]])]])
    return information


@dataclass
class Fit:
    estimates: dict
    variances: dict
    status: str
    gradient_max: float = np.nan
    information_condition: float = np.nan
    optimizer_message: str = ""


def survival_fit(y, delta, x, estimate_gamma=True):
    a = np.column_stack((np.ones(len(y)), x))
    p = a.shape[1]
    names = BETA_NAMES[:x.shape[1]] + (["gamma"] if estimate_gamma else [])
    empty = {name: np.nan for name in names}
    empty.update({name: np.nan for name, _, _ in profiles()})
    if delta.sum() == 0 or np.linalg.matrix_rank(a) < p:
        return Fit(empty, {}, "no_events_or_rank_deficient")
    # Start values do not use the true gamma. Retry only if needed.
    best = None
    for g0 in ([1.0, 2.0, 4.0] if estimate_gamma else [1.0]):
        intercept = np.log(delta.sum() / np.sum((y / SCALE_0) ** g0))
        start = np.r_[intercept, np.zeros(x.shape[1])]
        if estimate_gamma:
            start = np.r_[start, np.log(g0)]
        result = minimize(likelihood, start, args=(y, delta, a, estimate_gamma),
                          jac=True, method="BFGS",
                          options={"gtol": 1e-5, "maxiter": 500})
        grad = np.max(np.abs(result.jac))
        if np.isfinite(result.fun) and np.isfinite(result.x).all() and grad < 1e-3:
            best = result
            break
    if best is None:
        return Fit(empty, {}, "optimization_failed",
                   optimizer_message=str(result.message))
    par = best.x
    gamma = np.exp(par[-1]) if estimate_gamma else 1.0
    estimates = dict(zip(BETA_NAMES[:x.shape[1]], par[1:1+x.shape[1]]))
    if estimate_gamma:
        estimates["gamma"] = gamma
    for name, xp, _ in profiles():
        logmean = (np.log(SCALE_0) - (par[0] + xp[:x.shape[1]] @ par[1:1+x.shape[1]])
                   / gamma + gammaln(1 + 1 / gamma))
        estimates[name] = np.exp(logmean) if logmean < 700 else np.nan
    info = likelihood(par, y, delta, a, estimate_gamma, True)
    try:
        np.linalg.cholesky(info)
        condition = np.linalg.cond(info)
        if condition > 1e12:
            raise np.linalg.LinAlgError("ill-conditioned information")
        covariance = np.linalg.inv(info)
        diag = np.diag(covariance)[1:].copy()
        if estimate_gamma:
            diag[-1] *= gamma ** 2  # delta method from log gamma to gamma
        variances = {"model_based": dict(zip(names, diag))}
        return Fit(estimates, variances, "ok", float(np.max(np.abs(best.jac))),
                   float(condition), str(best.message))
    except np.linalg.LinAlgError:
        # Keep finite point estimates; flag missing uncertainty separately.
        return Fit(estimates, {}, "point_only_invalid_information",
                   float(np.max(np.abs(best.jac))), optimizer_message=str(best.message))


def ols_fit(y, x):
    a = np.column_stack((np.ones(len(y)), x))
    names = BETA_NAMES + [name for name, _, _ in profiles()]
    if np.linalg.matrix_rank(a) < a.shape[1]:
        return Fit(dict.fromkeys(names, np.nan), {}, "rank_deficient")
    fit = OLS(np.log(y), a).fit()
    estimates = dict(zip(BETA_NAMES, -GAMMA_TRUE * np.asarray(fit.params)[1:]))
    smear = np.mean(np.exp(fit.resid))
    for name, xp, _ in profiles():
        estimates[name] = np.exp(np.r_[1.0, xp] @ fit.params) * smear
    variances = {"model_based": dict(zip(
        BETA_NAMES, GAMMA_TRUE**2 * np.diag(fit.cov_params())[1:]))}
    return Fit(estimates, variances, "ok")


def summarize(raw):
    performance, calibration = [], []
    keys = ["design", "cutoff", "model", "parameter"]
    for key, group in raw.groupby(keys, sort=False):
        base = dict(zip(keys, key))
        valid = group.loc[np.isfinite(group.estimate)]
        est = valid.estimate.to_numpy()
        truth = float(group.truth.iloc[0])
        count = len(est)
        error = est - truth
        variance = np.var(est, ddof=1) if count > 1 else np.nan
        performance.append(base | {
            "truth": truth, "n_attempted": len(group), "n_successful": count,
            "failure_rate": 1-count/len(group),
            "mean_estimate": np.mean(est) if count else np.nan,
            "bias": np.mean(error) if count else np.nan,
            "variance": variance, "empirical_se": np.sqrt(variance),
            "mse": np.mean(error**2) if count else np.nan,
            "bias_mcse": np.sqrt(variance/count) if count else np.nan,
            "mse_mcse": np.std(error**2, ddof=1)/np.sqrt(count) if count > 1 else np.nan,
        })
        # Mean predictions have no analytical uncertainty estimate in this study.
        if key[-1].startswith("mean_T_"):
            continue
        for kind in COVARIANCES:
            col = "variance_" + kind
            paired = valid.loc[np.isfinite(valid[col]) & (valid[col] > 0)]
            m = len(paired)
            empirical = paired.estimate.var(ddof=1)
            reported = paired[col].mean()
            se = np.sqrt(paired[col])
            calibration.append(base | {
                "covariance": kind, "n_valid_variances": m, "n_attempted": len(group),
                "mean_estimated_variance": reported,
                "empirical_variance_same_replicates": empirical,
                "variance_ratio": reported/empirical if empirical > 0 else np.nan,
                "mean_estimated_se": se.mean(),
                "empirical_se_same_replicates": np.sqrt(empirical),
                "variance_estimate_mcse": paired[col].std(ddof=1)/np.sqrt(m) if m > 1 else np.nan,
            })
    return pd.DataFrame(performance), pd.DataFrame(calibration)


def make_plots(performance, calibration, out):
    """Plot every evaluated target; keep units separate and never pool errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Also support plotting existing summaries from earlier script versions.
    performance, calibration = performance.copy(), calibration.copy()
    for frame in (performance, calibration):
        frame["cutoff"] = pd.to_numeric(frame.cutoff, errors="coerce")
        frame.drop(frame.index[~np.isfinite(frame.cutoff)], inplace=True)
    calibration = calibration.loc[calibration.covariance == "model_based"]
    if performance.empty:
        raise ValueError("No finite censoring thresholds in the performance summary.")
    labels = {"correct_weibull": "Correct Weibull", "omitted_dirty_weibull": "Omit dirty air",
              "exponential": "Exponential", "naive_ols": "Naive OLS"}
    titles = {"beta_z": "Temperature coefficient", "beta_interaction": "Compound-temperature interaction",
              "beta_dirty": "Dirty-air coefficient", "gamma": "Weibull shape"}
    styles = dict(zip(MODELS, [
        dict(color="#1764A0", marker="o", linestyle="-"),
        dict(color="#C87516", marker="s", linestyle="--"),
        dict(color="#8A477D", marker="^", linestyle="-."),
        dict(color="#66752A", marker="D", linestyle=":"),
    ]))
    cutoffs = sorted(performance.cutoff.unique())
    metrics = [("bias", "Bias"), ("variance", "Empirical variance"), ("mse", "MSE")]
    mean_names = [name for name, _, _ in profiles()]
    handles = [Line2D([], [], label=labels[m], **styles[m]) for m in MODELS]

    def format_axis(ax, values, metric, ylabel):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if metric in ("variance", "mse") and len(finite) and np.all(finite > 0):
            ax.set_yscale("log")
            lo, hi = finite.min(), finite.max()
            ax.set_ylim(lo / 1.4, hi * 1.4)
            ylabel += " (log scale)"
        elif len(finite):
            lo, hi = min(0, finite.min()), max(0, finite.max())
            pad = 0.08 * (hi - lo) if hi > lo else 0.05
            ax.set_ylim(lo - pad, hi + pad)
        if metric == "bias":
            ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xticks(cutoffs, [f"{c:g}" for c in cutoffs])
        ax.set_xlabel("Censoring cutoff (laps)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)

    def plot_models(ax, frame, parameter, metric):
        for model in MODELS:
            g = frame.loc[(frame.model == model) & (frame.parameter == parameter)]
            g = g.set_index("cutoff").reindex(cutoffs)
            # Missing estimates remain gaps, rather than becoming zero errors.
            ax.plot(cutoffs, g[metric], linewidth=1.6, markersize=5, **styles[model])

    def save(fig, name, title, note):
        fig.suptitle(title, fontsize=15)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.948),
                   ncol=4, frameon=False)
        fig.text(0.5, 0.014, note, ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, 0.065, 1, 0.91))
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{name}.{extension}", dpi=180)
        plt.close(fig)

    for design in performance.design.unique():
        data = performance.loc[performance.design == design]
        attempts = data.n_attempted
        repetitions = (str(int(attempts.iloc[0])) if attempts.nunique() == 1
                       else f"{int(attempts.min())}-{int(attempts.max())}")
        subtitle = f"{design} covariates; {repetitions} repetitions"
        fig, axes = plt.subplots(4, 3, figsize=(13, 12))
        for row, parameter in enumerate(BETA_NAMES + ["gamma"]):
            for col, (metric, metric_title) in enumerate(metrics):
                ax = axes[row, col]
                plot_models(ax, data, parameter, metric)
                reference = performance.loc[performance.parameter == parameter, metric]
                format_axis(ax, reference, metric, metric_title)
                ax.set_title(f"{titles[parameter]} | {metric_title}", fontsize=10)
        save(fig, f"performance_{design}", f"All coefficient and shape estimates | {subtitle}",
             "Scales differ between parameters; corresponding panels match across covariate designs.\n"
             "Dirty-air omission has no dirty-air estimate. Shape is estimated only by the Weibull models.\n"
             "Summaries use available estimates; see CSV files for success counts and Monte Carlo standard errors.")

        for metric, metric_title in metrics:
            fig, axes = plt.subplots(2, 3, figsize=(13, 8))
            reference = performance.loc[performance.parameter.isin(mean_names), metric]
            for row, compound in enumerate((0, 1)):
                for col, temperature in enumerate((25, 35, 45)):
                    name = f"mean_T_t{temperature}_c{compound}_d0.5"
                    ax = axes[row, col]
                    plot_models(ax, data, name, metric)
                    units = "laps" if metric == "bias" else "laps\u00b2"
                    axis_title = "Variance" if metric == "variance" else metric_title
                    format_axis(ax, reference, metric, f"{axis_title} ({units})")
                    compound_label = "intermediate" if compound == 0 else "hard"
                    ax.set_title(f"{temperature} C | {compound_label} compound", fontsize=11)
            save(fig, f"mean_lifetime_{metric}_{design}",
                 f"Mean lifetime: all six profiles | {metric_title} | {subtitle}",
                 "Dirty air = 0.5 in every profile. Scales match across all profiles and both covariate designs.\n"
                 "Profiles are separate conditional means, not a population-wide average. Success counts and MCSEs: CSV files.")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
        for ax, parameter in zip(axes, BETA_NAMES):
            rows = calibration.loc[(calibration.model == "naive_ols")
                                   & (calibration.parameter == parameter)]
            g = rows.loc[rows.design == design].set_index("cutoff").reindex(cutoffs)
            ax.plot(cutoffs, g.variance_ratio, linewidth=1.6, **styles["naive_ols"])
            ratio_reference = calibration.loc[(calibration.model == "naive_ols")
                                               & calibration.parameter.isin(BETA_NAMES), "variance_ratio"]
            format_axis(ax, np.r_[ratio_reference, 1], "ratio", "Estimated / empirical variance")
            ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
            ax.set_title(titles[parameter], fontsize=10)
        fig.suptitle(f"Classical OLS variance accuracy: all slopes | {subtitle}", fontsize=14)
        fig.text(0.5, 0.025,
                 "Dashed line = agreement (1). Below 1: underestimated variability; above 1: overestimated variability.\n"
                 "Model-based and empirical variances use matching valid repetitions. See variance_diagnostics.csv for counts.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.13, 1, 0.92))
        for extension in ("png", "pdf"):
            fig.savefig(out / f"ols_calibration_{design}.{extension}", dpi=180)
        plt.close(fig)


def run(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}. Choose a new --out.")
    out.mkdir(parents=True, exist_ok=True)
    designs = ["random", "fixed"] if args.design == "both" else [args.design]
    cutoffs = [float(c) for c in args.cutoffs]
    rows, censor_rows, fit_rows, fixed_rows = [], [], [], []
    truth = PARAM_TRUTH | {name: t for name, _, t in profiles()}
    for design in designs:
        # Random mode preserves original RNG ordering for the original cutoffs.
        rng = np.random.default_rng(args.seed)
        fixed = covariates(np.random.default_rng(args.seed + 1), args.races, args.stints)
        if design == "fixed":
            for i, (x, race_id) in enumerate(zip(*fixed)):
                fixed_rows.append({"stint": i, "race": race_id,
                                   "z": x[0], "compound_z": x[1], "dirty": x[2]})
        for rep in range(1, args.reps + 1):
            x, race = covariates(rng, args.races, args.stints) if design == "random" else fixed
            t = failure_times(rng, x)
            for cutoff in cutoffs:
                label = f"{cutoff:g}"
                y, delta = np.minimum(t, cutoff), (t <= cutoff).astype(float)
                base = {"design": design, "repetition": rep, "cutoff": label}
                censor_rows.append(base | {"censoring_proportion": np.mean(delta == 0),
                                           "n_events": int(delta.sum())})
                fits = {
                    "correct_weibull": survival_fit(y, delta, x),
                    "omitted_dirty_weibull": survival_fit(y, delta, x[:, :2]),
                    "exponential": survival_fit(y, delta, x, False),
                    "naive_ols": ols_fit(y, x),
                }
                for model, fit in fits.items():
                    fit_rows.append(base | {"model": model, "status": fit.status,
                                            "gradient_max": fit.gradient_max,
                                            "information_condition": fit.information_condition,
                                            "optimizer_message": fit.optimizer_message})
                    for parameter, estimate in fit.estimates.items():
                        row = base | {"model": model, "parameter": parameter,
                                      "truth": truth[parameter], "estimate": estimate}
                        for kind in COVARIANCES:
                            row["variance_"+kind] = fit.variances.get(kind, {}).get(parameter, np.nan)
                        rows.append(row)
            if rep % max(1, args.reps // 20) == 0:
                print(f"{design}: {rep}/{args.reps} replicates", flush=True)
    raw = pd.DataFrame(rows)
    performance, calibration = summarize(raw)
    censor = pd.DataFrame(censor_rows)
    censor_summary = censor.groupby(["design", "cutoff"], sort=False).agg(
        mean_censoring_proportion=("censoring_proportion", "mean"),
        sd_censoring_proportion=("censoring_proportion", "std"),
        mean_events=("n_events", "mean"), n_repetitions=("n_events", "size")).reset_index()
    censor_summary["censoring_mcse"] = (censor_summary.sd_censoring_proportion
                                        / np.sqrt(censor_summary.n_repetitions))
    fitlog = pd.DataFrame(fit_rows)
    outputs = {
        "raw_estimates.csv": raw, "performance_summary.csv": performance,
        "variance_diagnostics.csv": calibration, "censoring_summary.csv": censor_summary,
        "prediction_profiles.csv": pd.DataFrame([
            {"parameter": name, "z": x[0], "compound_z": x[1], "dirty": x[2], "true_mean_laps": t}
            for name, x, t in profiles()]),
    }
    if fixed_rows:
        outputs["fixed_covariates.csv"] = pd.DataFrame(fixed_rows)
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    if not args.no_plots:
        make_plots(performance, calibration, out)
    print(f"Saved outputs to {out.resolve()}", flush=True)
    print(fitlog.groupby(["design", "status"]).size().to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--design", choices=["random", "fixed", "both"], default="random")
    parser.add_argument("--out", default="f1_results_censored")
    parser.add_argument("--races", type=int, default=20)
    parser.add_argument("--stints", type=int, default=60)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[15, 20, 25])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.reps < 2 or args.races < 3 or args.stints < 2:
        parser.error("Use at least 2 repetitions, 3 races, and 2 stints per race.")
    if args.seed < 0 or any(not np.isfinite(c) or c <= 0 for c in args.cutoffs):
        parser.error("Seed must be nonnegative and cutoffs finite and positive.")
    if len(set(args.cutoffs)) != len(args.cutoffs):
        parser.error("Cutoffs must be unique.")
    run(args)


if __name__ == "__main__":
    main()
