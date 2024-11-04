import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from utils import feature_selection, column_names

sns.set_theme()

# Parse the arguments
parser = argparse.ArgumentParser(description="Random Forest model for aquifers")
parser.add_argument(
    "--target",
    type=str,
    default="BFI",
    help="The target variable to predict",
)
parser.add_argument(
    "--bootstrap-repeats",
    type=int,
    default=10000,
    help="Number of bootstrap repeats",
)

args = parser.parse_args()

target = args.target
m = args.bootstrap_repeats

# Load the dataset
df = pd.read_csv("data/aquifers.csv")
names = df.pop("AQUIFER")
y = df[target]
X = df[column_names]
# feature selection
columns = feature_selection(X, y)
X = X[columns]


# Define the model
# per sample squared error
def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


model_kwargs = dict(n_estimators=100, oob_score=squared_error)

# Train the model
scores = []
scores_se = []
baselines = []
baselines_se = []
for i in range(2, len(X), 1):
    score = []
    baseline = []
    linear = []
    for j in tqdm(range(m), desc=f"{i:2}", leave=False):
        np.random.seed(i * m + j)
        ind = np.random.choice(X.index, size=i, replace=True)
        X_boot = X.loc[ind]
        y_boot = y.loc[ind]
        # random forest
        clf = RandomForestRegressor(random_state=j, **model_kwargs)
        clf.fit(X_boot, y_boot)
        # Validation
        ind_val = np.setdiff1d(X.index, ind)
        X_val = X.loc[ind_val]
        y_val = y.loc[ind_val]
        s = np.square(y_val - clf.predict(X_val))
        b = np.square(y_val - np.ones_like(y_val) * y_boot.mean())
        rmse = np.sqrt(s.mean())
        baseline_rmse = np.sqrt(b.mean())
        score.append(rmse)
        baseline.append(baseline_rmse)
    print(
        f"{i:2} {np.mean(score) : 0.4} {np.std(score) : 0.3} | {np.mean(baseline) : 0.4} {np.std(baseline) : 0.3}"
    )
    scores.append(np.mean(score))
    scores_se.append(np.std(score) / np.sqrt(len(score)))
    baselines.append(np.mean(baseline))
    baselines_se.append(np.std(baseline) / np.sqrt(len(baseline)))


df = pd.DataFrame(
    {
        "Dataset size": range(2, len(scores) + 2),
        "Random Forest": scores,
        "Random Forest SE": scores_se,
        "Baseline": baselines,
        "Baseline SE": baselines_se,
    }
)

df.to_csv(f"results/dataset_size_{target}.csv", index=False)

# Visualize the results
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.errorbar(
    range(2, len(scores) + 2), scores, yerr=np.array(scores_se).T, label="Random Forest"
)
ax.errorbar(
    range(2, len(scores) + 2),
    baselines,
    yerr=np.array(baselines_se).T,
    label="Average",
    alpha=0.6,
)
ax.set_xlabel("Dataset size")
ax.set_ylabel("RMSE")

ax.legend()
ax.set_title("Model error as a function of dataset size")
fig.tight_layout()
plt.savefig(f"figures/dataset_size_{target}.png", dpi=300)
