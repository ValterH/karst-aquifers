import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from alibi.explainers import ALE, PartialDependence, plot_ale
from sklearn.inspection import PartialDependenceDisplay

from utils import feature_selection, column_names, y_columns

sns.set_theme()

os.makedirs("results/ALE", exist_ok=True)
os.makedirs("results/PDP", exist_ok=True)
os.makedirs("results/ICE", exist_ok=True)

df = pd.read_csv("data/aquifers.csv")
names = df.pop("AQUIFER")


X = df[column_names]

model_kwargs = dict(n_estimators=100)


dci_df = pd.read_csv("results/drop_column_importance.csv")
dci_df.set_index("Variable", inplace=True)

for quantity in y_columns:
    X = df[column_names]
    y = df[quantity]

    # feature selection
    columns = feature_selection(
        X, y, model_cls=RandomForestRegressor, model_kwargs=model_kwargs
    )
    X = X[columns]

    feature_importances = dci_df.loc[quantity][column_names[1:]]
    best_features = feature_importances.dropna().sort_values(ascending=False).index

    model = RandomForestRegressor(random_state=42, **model_kwargs)
    model.fit(X[best_features].values, y)

    ale = ALE(model.predict, feature_names=best_features)
    exp_ale = ale.explain(X[best_features].values)

    pdp = PartialDependence(model.predict, feature_names=best_features)
    exp_pdp = pdp.explain(X[best_features].values, kind="both")

    df_ale = pd.DataFrame()
    for i, feature in enumerate(exp_ale.data["feature_names"]):
        new_columns = pd.DataFrame(
            {
                f"{feature}_ale": exp_ale.data["ale_values"][i].flatten(),
                f"{feature}_feature": exp_ale.data["feature_values"][i],
            }
        )
        df_ale = pd.concat([df_ale, new_columns], axis=1)
    df_ale["constant"] = np.ones(df_ale.shape[0]) * exp_ale.data["constant_value"]

    df_pdp = pd.DataFrame()
    df_ice = pd.DataFrame()
    for i, feature in enumerate(exp_pdp.data["feature_names"]):
        pd_values = exp_pdp.data["pd_values"][i].flatten()
        feature_values = exp_pdp.data["feature_values"][i]
        new_columns = pd.DataFrame(
            {
                f"{feature}_pd": pd_values,
                f"{feature}_feature": feature_values,
            }
        )
        df_pdp = pd.concat([df_pdp, new_columns], axis=1)

        ice_values = exp_pdp.data["ice_values"][i].squeeze()
        for j, name in enumerate(names):
            ice_feature = ice_values[j]
            new_columns = pd.DataFrame(
                {
                    f"{feature}_{name}_ice": ice_feature,
                }
            )
            df_ice = pd.concat([df_ice, new_columns], axis=1)

    df_ale.to_csv(f"results/ALE/{quantity}_ale.csv", index=False)
    df_pdp.to_csv(f"results/PDP/{quantity}_pdp.csv", index=False)
    df_ice.to_csv(f"results/ICE/{quantity}_ice.csv", index=False)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_ale(exp_ale, ax=ax, n_cols=4)
    fig.tight_layout()
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is None:
            continue
        legend.remove()
    plt.savefig(f"figures/ALE/{quantity}_ale.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 8))
    pdp = PartialDependenceDisplay.from_estimator(
        model,
        X[best_features].values,
        features=np.arange(len(best_features)),
        feature_names=best_features,
        kind="both",
        ax=ax,
        n_cols=4,
    )
    fig.tight_layout()
    plt.savefig(f"figures/PDP/{quantity}_pdp.png")
    plt.close()
