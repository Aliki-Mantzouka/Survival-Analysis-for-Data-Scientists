___
1. performance_summary.csv: the main results table.
	- This contains results for each combination of covariate design (random or fixed), censoring cutoff, model, estimated quantity: regression coefficients, estimated Weibull shape, and mean lifetime; and respective performance measurements: bias, variance, empirical standard error, MSE, and Monte Carlo standard errors (`*_mcse` of the bias and the MSE.
2. variance_diagnostics.csv
	- compares the mean variance reported by each fitted model with the empirical variation actually observed across repeated datasets.
	- presents per combination of covariate design, cutoff level, model and parameter, it's covariance, mean estimated variance, empirical variance same replicates and variance ratio
	- How to interpret the ratio:
		- Near 1: the two variance measures agree.
		- Below 1: the model underestimates variability.
		- Above 1: the model overestimates variability.
	- **Report purpose is to answer teacher's concern 2:** use the `naive_ols` coefficient rows to answer whether classical OLS variance estimates remain accurate when censored replacements are treated as failures.
3. censoring_summary.csv
	- This summarizes the average proportion censored and the average number of observed failures for each cutoff and design.
	- Can be used in the report to translate “15-, 20-, and 25-lap cutoffs” into the amount of information actually observed.
4. performance_random.png
	- Main visual summary of the performance measurements across different censoring levels for resampled covariates
5. ols_calibration_random.png
	- Nice picture to use as a visual answer to teacher's concern 2
	- show whether OLS underestimates or overestimates coefficient variability and how that changes with censoring
6. raw_estimates.csv
	- This stores each individual estimate, its true value, and its model-based variance for every repetition made -> ignore for the report
7. prediction_profiles.csv
	- This records the covariate values defining each prediction profile and its true conditional mean lifetime.
8. fixed_covariates.csv
	- This records the covariates used for each stint and race in the fixed-design study. -> ignore for the report