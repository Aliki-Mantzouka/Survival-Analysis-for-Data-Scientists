import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

SEED = 12345
R = 2000 # number of repetitions
N_RACES = 20 # number of races per season
STINTS_PER_RACE = 60 # number of stints per race
GAMMA_TRUE = 4.0 # true shape parameter of the Weibull distribution
SCALE_0 = 30.0
Q_20 = 26.58
BETA_TRUE = {"beta_z": 0.6, "beta_interaction": -0.4, "beta_dirty": 0.8}
TRUTH_PARAMS = {**BETA_TRUE, "gamma": GAMMA_TRUE}
C_VALUES = [15, 20, 25] # different censoring levels set for the simulation


def simulate_latent_data(rng): # this funtion generates the data for one reepetiton.rng is the random number generator

    race_temp = rng.normal(35, 10, size=N_RACES)
    temperature = np.repeat(race_temp, STINTS_PER_RACE)
    n = N_RACES * STINTS_PER_RACE

    compound = rng.binomial(1, 0.8, size=n)  # 1 = hard, 0 = intermediate
    dirty = rng.beta(0.5, 0.4, size=n)

    z = (temperature - Q_20) / 10
    eta = (BETA_TRUE["beta_z"] * z
           + BETA_TRUE["beta_interaction"] * compound*z
           + BETA_TRUE["beta_dirty"] * dirty)
    scale = SCALE_0 * np.exp(-eta / GAMMA_TRUE)

    u = rng.uniform(0, 1, size=n)
    t = scale * (-np.log(u)) ** (1 / GAMMA_TRUE)

    return z, compound*z, dirty, t


def neg_log_likelihood(params, y, delta, X, estimate_gamma):

    # definition of parameters from the Weibull cumulative hazard function, used for the log-likelihood calculation
    alpha = params[0]
    beta = params[1:1 + X.shape[1]]
    gamma = np.exp(params[-1]) if estimate_gamma else 1.0 

    lin = alpha + X @ beta
    loglik = np.sum(delta * (np.log(gamma) + (gamma - 1) * np.log(y) + lin)
                    - np.exp(lin) * y ** gamma)
    
    return -loglik


def neg_log_likelihood_gradient(params, y, delta, X, estimate_gamma): # derivative of the negative log-likelihood with respect to the parameters
    
    alpha = params[0]
    beta = params[1:1 + X.shape[1]]
    gamma = np.exp(params[-1]) if estimate_gamma else 1.0 # we use the exponential funtion here because params[-1] is the logarithm of gamma when estimate_gamma is True, since we useSciPy's unrestricted optimizater (all possible values) and we want gamma to always be positive

    cumulative_hazard = np.exp(alpha + X @ beta) * y ** gamma
    residual = delta - cumulative_hazard
    grad = np.concatenate([[residual.sum()], X.T @ residual])
    if estimate_gamma:
        d_gamma = np.sum(delta * (1 / gamma + np.log(y)) - cumulative_hazard * np.log(y))
        grad = np.append(grad, d_gamma * gamma) # since we are optimizing the logarithm of gamma instead of gamma itself, we multiply by gamma to get the correct gradient (chain rule)
    return -grad


def fit_survival_model(y, delta, X, estimate_gamma):

    p = X.shape[1]

    if estimate_gamma:
        start = np.concatenate([[-GAMMA_TRUE * np.log(SCALE_0)], np.zeros(p), [np.log(GAMMA_TRUE)]]) # again, here we are initializing the starting value for the logarithm of gamma instead of gamma itself
    else:
        start = np.concatenate([[-np.log(SCALE_0)], np.zeros(p)])

    with np.errstate(over="ignore", invalid="ignore"):
        result = minimize(neg_log_likelihood, start, args=(y, delta, X, estimate_gamma), # minimize() sets the `params` parameter as the `start` vector
                          jac=neg_log_likelihood_gradient, method="BFGS", options={"gtol": 1e-3}) # BFGS is SciPy's optimization method, that is free to try any real number, that is why we optimize the log of gamma
        
    if not result.success or not np.all(np.isfinite(result.x)):
        return None

    return result.x


def fit_ols(y, X): # seperate function to fit the naive OLS regression model

    fit = sm.OLS(np.log(y), sm.add_constant(X)).fit() # we fit the OLS estimator on the log of T, because given the format of the chosen time generation expression, the log of T is approximately linear in the covariates
    return -4 * np.asarray(fit.params)[1:]


def fit_all_models(y, delta, z, interaction, dirty):

    X3 = np.column_stack([z, interaction, dirty]) # full model with all covariates
    X2 = np.column_stack([z, interaction])  # model omitting the dirty covariate
    names3 = ["beta_z", "beta_interaction", "beta_dirty"]
    names2 = ["beta_z", "beta_interaction"]

    out = {}
    for model, X, names, est_gamma in [
        ("correct_weibull", X3, names3, True),
        ("omitted_dirty_weibull", X2, names2, True),
        ("exponential", X3, names3, False),
    ]:
        params = fit_survival_model(y, delta, X, est_gamma)
        all_names = names + (["gamma"] if est_gamma else [])
        if params is None:
            values = [np.nan] * len(all_names)
        else:
            values = list(params[1:1 + X.shape[1]])
            if est_gamma:
                values.append(np.exp(params[-1]))
        out[model] = (all_names, values)

    out["naive_ols"] = (names3, list(fit_ols(y, X3)))
    return out


def summarize(raw):
    rows = []
    for (C, model, parameter), g in raw.groupby(["censoring_threshold", "model", "parameter"]):
        est = g["estimate"].dropna()
        truth = TRUTH_PARAMS[parameter]
        rows.append({
            "censoring_threshold": C,
            "model": model,
            "parameter": parameter,
            "n_successful": len(est),
            "mean_estimate": est.mean(),
            "bias": est.mean() - truth,
            "variance": est.var(ddof=1),
            "mse": np.mean((est - truth) ** 2),
        })
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED) # NumpPy's random data generator
    rows = []
    cens_rows = []

    for rep in range(1, R + 1):
        z, interaction, dirty, t = simulate_latent_data(rng) # generate data
        for C in C_VALUES: # for each censoring threshold
            y = np.minimum(t, C) # observed time, censored at C
            delta = (t <= C).astype(float) # number of uncensored observations
            cens_rows.append({"repetition": rep, "censoring_threshold": C,
                              "censoring_proportion": np.mean(delta == 0)})
            for model, (names, values) in fit_all_models(y, delta, z, interaction, dirty).items(): # fit all specified models, for this given generated dataset, for this censoring threshold
                for name, value in zip(names, values):
                    rows.append({"repetition": rep, "censoring_threshold": C, "model": model,
                                    "parameter": name, "estimate": value})
    raw = pd.DataFrame(rows)
    cens = pd.DataFrame(cens_rows)

    performance = summarize(raw)
    censoring_summary = (cens.groupby("censoring_threshold")["censoring_proportion"]
                         .mean().rename("mean_censoring_proportion").reset_index())

    raw.to_csv("raw_estimates.csv", index=False)
    performance.to_csv("performance_summary.csv", index=False)
    censoring_summary.to_csv("censoring_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print(performance.to_string(index=False))
    print()
    print(censoring_summary.to_string(index=False))


if __name__ == "__main__":
    main()


