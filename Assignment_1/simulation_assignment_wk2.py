import numpy as np
import matplotlib.pyplot as plt

np.random.seed(6767)

# sample size
n = 500
# true lambda
lambda_ = 0.05
# censoring percentage set to 0%, 10% and 50% for the task
p_censor = 0.1
# number of repetitions
r = 2000


censoring_levels = [0.0, 0.1, 0.5] # the sensoring 

results = []

for p_censor in censoring_levels:

    lambdas_MLE = np.zeros(r)
    lambdas_naive = np.zeros(r)

    zero_event_count = 0

    for i in range(r): # for each of the sensoring levels

        # true survival times

        true_times = np.random.exponential(scale=1 / lambda_, size=n)

        if p_censor == 0.0: # we simulate that this is the naive state
            
            observed_times = true_times
            delta = np.ones(n, dtype=int)
        else:
            lambda_c = lambda_ * p_censor / (1 - p_censor)
            censored_times = np.random.exponential(scale=1 / lambda_c, size=n)

            # from them get Y_i = min(T_i, C_i) and delta_i = I(T_i <= C_i)
            observed_times = np.minimum(true_times, censored_times)
            
            delta = (true_times <= censored_times).astype(int)

        # calculate MLE and naive estimates of lambda
        if delta.sum() == 0:
            zero_event_count += 1
            lambdas_MLE[i] = np.nan
        else:
            lambdas_MLE[i] = delta.sum() / observed_times.sum()

        lambdas_naive[i] = 1 / observed_times.mean()

    zero_event_proportion = zero_event_count / r

    # calculate bias, variance and MSE for both estimators. we use nan-aware functions for MLE to handle zero-event cases.
    bias_MLE = np.nanmean(lambdas_MLE) - lambda_
    bias_naive = np.mean(lambdas_naive) - lambda_

    var_MLE = np.nanvar(lambdas_MLE, ddof=1)
    var_naive = np.var(lambdas_naive, ddof=1)

    mse_MLE = np.nanmean((lambdas_MLE - lambda_) ** 2)
    mse_naive = np.mean((lambdas_naive - lambda_) ** 2)

    print(f'Number of zero-event simulations for censoring level {p_censor}: {zero_event_count}')

    results.append({
        "censoring": p_censor,
        "zero_event_proportion": zero_event_proportion,
        "bias_MLE": bias_MLE,
        "bias_naive": bias_naive,
        "var_MLE": var_MLE,
        "var_naive": var_naive,
        "mse_MLE": mse_MLE,
        "mse_naive": mse_naive,
        "lambdas_MLE": lambdas_MLE,
        "lambdas_naive": lambdas_naive,
    })

header = f"{'Censor%':>8} | {'Zero Event%':>11} | {'Bias MLE':>11} | {'Bias Naive':>11} | {'Var MLE':>11} | {'Var Naive':>11} | {'MSE MLE':>11} | {'MSE Naive':>11}"
print(header)
print("-" * len(header))
for res in results:
    print(f"{res['censoring']*100:7.0f}% | "
          f"{res['zero_event_proportion']*100:11.6f} | "
          f"{res['bias_MLE']:11.6f} | {res['bias_naive']:11.6f} | "
          f"{res['var_MLE']:11.6f} | {res['var_naive']:11.6f} | "
          f"{res['mse_MLE']:11.6f} | {res['mse_naive']:11.6f}")



censor_pct = [res["censoring"] * 100 for res in results]

bias_mle_vals = [res["bias_MLE"] for res in results]
bias_naive_vals = [res["bias_naive"] for res in results]

var_mle_vals = [res["var_MLE"] for res in results]
var_naive_vals = [res["var_naive"] for res in results]

mse_mle_vals = [res["mse_MLE"] for res in results]
mse_naive_vals = [res["mse_naive"] for res in results]


# Here we are plotting the difference between the MLE approach and the naive approach

plt.figure(figsize=(7, 5))
plt.plot(censor_pct, bias_mle_vals, marker="o", label="MLE")
plt.plot(censor_pct, bias_naive_vals, marker="o", label="Naive")
plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
plt.xlabel("Censoring level (%)")
plt.ylabel("Bias")
plt.title("Bias vs. Censoring Level")
plt.legend()
plt.tight_layout()
plt.savefig("bias_vs_censoring.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(censor_pct, var_mle_vals, marker="o", label="MLE")
plt.plot(censor_pct, var_naive_vals, marker="o", label="Naive")
plt.xlabel("Censoring level (%)")
plt.ylabel("Variance")
plt.title("Variance vs. Censoring Level")
plt.legend()
plt.tight_layout()
plt.savefig("variance_vs_censoring.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(censor_pct, mse_mle_vals, marker="o", label="MLE")
plt.plot(censor_pct, mse_naive_vals, marker="o", label="Naive")
plt.yscale("log")
plt.xlabel("Censoring level (%)")
plt.ylabel("MSE (log scale)")
plt.title("MSE vs. Censoring Level")
plt.legend()
plt.tight_layout()
plt.savefig("mse_vs_censoring.png", dpi=150)
plt.close()


print("\nPlots saved: bias_vs_censoring.png, variance_vs_censoring.png, "
      "mse_vs_censoring.png")