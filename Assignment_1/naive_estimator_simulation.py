import numpy as np
import pandas as pd

np.random.seed(6767)

def run_naive_simulation(n=500, lambda_true=0.05, r_reps=2000, censoring_proportions=[0.0, 0.1, 0.5]):
    results = []

    for p_censor in censoring_proportions:
        lambdas_naive = np.zeros(r_reps)

        for i in range(r_reps):
            # generate true failure times: T ~ Exp(lambda)
            true_times = np.random.exponential(scale=1.0 / lambda_true, size=n)

            # apply censoring mechanism
            if p_censor == 0.0:
                # no censoring: observed time Y = T
                observed_times = true_times
            else:
                # calculate censoring rate parameter (rho) to achieve target censoring proportion
                rho = (lambda_true * p_censor) / (1.0 - p_censor)
                censoring_times = np.random.exponential(scale=1.0 / rho, size=n)
                
                # observed time is the minimum: Y = min(T, C)
                observed_times = np.minimum(true_times, censoring_times)

            # naive estimator: lambda_hat_naive = 1 / mean(Y)
            lambdas_naive[i] = 1.0 / np.mean(observed_times)

        # calculate Monte Carlo metrics
        bias = np.mean(lambdas_naive) - lambda_true
        variance = np.var(lambdas_naive, ddof=1)
        mse = np.mean((lambdas_naive - lambda_true) ** 2)

        results.append({
            "Censoring (%)": f"{int(p_censor * 100)}%",
            "True Lambda": lambda_true,
            "Mean Estimate": np.mean(lambdas_naive),
            "Bias": bias,
            "Variance": variance,
            "MSE": mse
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Running Naive Estimator Simulation...\n")
    df_results = run_naive_simulation()
    
    # display results
    print(df_results.to_string(index=False))