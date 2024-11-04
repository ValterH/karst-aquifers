import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ttest_1samp

from utils import feature_selection, column_names, y_columns


def error_estimation(X, y, cv_repetitions=10):
    # compute the baseline (mean) error
    baseline = []
    for i, val in enumerate(y):
        baseline.append(np.mean(y[:i].tolist() + y[i + 1 :].tolist()))

    baseline_error = (y - baseline) ** 2

    # repeat cross-validation m times
    mse = []

    for i in tqdm(range(cv_repetitions)):
        clf = RandomForestRegressor(**model_kwargs, random_state=i)
        mse.append(
            -cross_val_score(clf, X, y, cv=len(X), scoring="neg_mean_squared_error")
        )

    # print the MSE and the confidence interval results
    mse = np.array(mse)
    t_1samp = ttest_1samp(mse.mean(axis=1), baseline_error.mean(), alternative="less")
    return mse, t_1samp.pvalue, baseline_error.values


df = pd.read_csv("data/aquifers.csv")
names = df.pop("AQUIFER")

model_kwargs = dict(n_estimators=100)

results = dict()

for quantity in y_columns:
    print(f"Estimating the error for {quantity}")
    X = df[column_names]
    y = df[quantity]

    # feature selection
    columns = feature_selection(
        X, y, model_cls=RandomForestRegressor, model_kwargs=model_kwargs
    )
    X = X[columns]

    mse, pvalue, baseline_error = error_estimation(X, y, cv_repetitions=10)

    mse_mean = mse.mean(axis=0)
    idx = mse_mean.argsort()
    mse_se = mse.flatten().std(axis=0) / np.sqrt(len(mse.flatten()))

    print(f"Y - {quantity.upper()}, E[Y]={y.mean():.3}, Var[Y]={y.std()**2 :.3}")
    print(f'{"Name":<35} {"MSE":>10} : 95% CI')
    for i in idx:
        name = names[i]
        # consider the 95% confidence interval using quantiles
        ci_low = np.quantile(mse[:, i], 0.025)
        ci_high = np.quantile(mse[:, i], 0.975)
        print(
            f'{name:<36} {f"{mse_mean[i] : 0.3}" :<9} : [{f"{ci_low : 0.3}" :<9}, {f"{ci_high : 0.3}" :<9}]'
        )
    print("-" * 72)
    print(
        f'{"Combined":<36} {f"{mse_mean.mean() : 0.3}" :<9} : [{f"{np.quantile(mse.flatten(), 0.025) : 0.3}" :<9}, {f"{np.quantile(mse.flatten(), 0.975) : 0.3}" :<9}]'
    )
    print()
    print("MSE(θ) < MSE(μ)")
    print(f"T-test: p={pvalue :.4}")

    # # RMSE +- SE
    rmse = np.sqrt(mse.mean(axis=1))
    print(f"Expected RMSE = {rmse.mean():.3} ± {rmse.std() / np.sqrt(len(rmse)):.3}")
    print(f"Baseline RMSE = {np.sqrt(baseline_error.mean()) :.3}")
    print("-" * 72)

    quantity_name = y_columns[quantity]
    results[quantity_name] = {
        "RMSE": rmse.mean(),
        "RMSE_SE": rmse.std() / np.sqrt(len(rmse)),
        "pvalue": pvalue,
        "baseline RMSE": np.sqrt(baseline_error.mean()),
    }

# save the results
df = pd.DataFrame(results).T
df.reset_index(inplace=True)
df.rename(columns={"index": "Variable"}, inplace=True)
df.to_csv("results/estimated_model_error.csv", index=False)
