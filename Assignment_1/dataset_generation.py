import numpy as np

np.random.seed(6767)

# sample size
n = 500
# true lambda
lambda_ = 0.05
# censoring percentage set to 0%, 10% and 50% for the task
p_censor = 0.1
# number of repetitions
r = 2000

# true survival times
true_times = np.random.exponential(scale=1/lambda_, size=n)

lambdas_MLE = np.zeros(r)
lambdas_naive = np.zeros(r)

for i in range(r):

    # generate censored times
    censored_times = np.random.exponential(scale=1/(lambda_*p_censor/(1-p_censor)), size=n)
    # from them get Y_i = min(T_i, C_i) and delta_i = I(T_i <= C_i)
    observed_times = np.minimum(true_times, censored_times)
    delta = (true_times <= censored_times).astype(int)

    # calculate MLE and naive estimates of lambda
    lambdas_MLE[i] = delta.sum() / observed_times.sum()
    lambdas_naive[i] = 1 / observed_times.mean()

# calculate bias, variance and MSE for both estimators
bias_MLE = np.mean(lambdas_MLE) - lambda_
bias_naive = np.mean(lambdas_naive) - lambda_

var_MLE = 1/(r-1) * np.sum((lambdas_MLE - np.mean(lambdas_MLE))**2)
var_naive = 1/(r-1) * np.sum((lambdas_naive - np.mean(lambdas_naive))**2)

mse_MLE = 1/r * np.sum((lambdas_MLE - lambda_)**2)
mse_naive = 1/r * np.sum((lambdas_naive - lambda_)**2)

print(f"Bias of MLE: {bias_MLE:.4e}\n")
print(f"Bias of Naive: {bias_naive:.4e}\n")   
print(f"Variance of MLE: {var_MLE:.4e}\n")
print(f"Variance of Naive: {var_naive:.4e}\n")
print(f"MSE of MLE: {mse_MLE:.4e}\n")
print(f"MSE of Naive: {mse_naive:.4e}\n")