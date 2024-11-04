import numpy as np
import pandas as pd
from tqdm import tqdm
from rfpimp import oob_dropcol_importances
from sklearn.ensemble import RandomForestRegressor

from utils import negative_rmse, feature_selection, column_names, y_columns


def feature_importance(X, y, m=100, description=""):
    drop_column_importances = []
    model = RandomForestRegressor(**model_kwargs)
    model.fit(X, y)
    dropcolumn_importance = oob_dropcol_importances(model, X, y, metric=negative_rmse)
    for i in tqdm(range(m), desc=description):
        np.random.seed(i)
        ind = np.random.choice(X.index, size=len(X), replace=True)
        X_boot = X.loc[ind]
        y_boot = y.loc[ind]
        model = RandomForestRegressor(**model_kwargs, random_state=i)

        drop_imp = oob_dropcol_importances(model, X_boot, y_boot, metric=negative_rmse)
        drop_imp.sort_index(inplace=True)
        drop_column_importances.append(drop_imp.values[:, 0])

    drop_column_importances = np.array(drop_column_importances)
    features = drop_imp.index.values
    assert (features == drop_imp.index.values).all()
    return (
        drop_column_importances,
        features,
        dropcolumn_importance.loc[features].Importance.values,
    )


df = pd.read_csv("data/aquifers.csv")


m = 100  # number of bootstrap repetitions


model_kwargs = dict(n_estimators=100)

results = dict()

for quantity in y_columns:
    X = df[column_names]
    y = df[quantity]

    # feature selection
    columns = feature_selection(
        X, y, model_cls=RandomForestRegressor, model_kwargs=model_kwargs
    )
    X = X[columns]

    # estimate the feature importance
    feature_importances, features, dropcolumn_importance = feature_importance(
        X, y, m=m, description=quantity
    )
    quantity_results = {
        feature: importance
        for feature, importance in zip(features, dropcolumn_importance)
    }

    for i, feature in enumerate(features):
        quantity_results[f"{feature}_SE"] = feature_importances[:, i].std() / np.sqrt(m)

    results[quantity] = dict(sorted(quantity_results.items()))

# save the results
df = pd.DataFrame(results).T
df.reset_index(inplace=True)
df.rename(columns={"index": "Variable"}, inplace=True)
df.to_csv("results/drop_column_importance.csv", index=False)
