import numpy as np
from rfpimp import oob_dropcol_importances
from sklearn.metrics import mean_squared_error


def negative_rmse(y_true, y_pred):
    return -np.sqrt(mean_squared_error(y_true, y_pred))


def feature_selection(X, y, model_cls, model_kwargs=dict(n_estimators=100)):
    columns = X.columns.tolist()

    while len(columns) > 2:
        X = X[columns]
        model = model_cls(**model_kwargs)
        importances = oob_dropcol_importances(model, X, y, metric=negative_rmse)
        negative_importance_features = importances[importances.Importance < 0]
        if len(negative_importance_features):
            worst_feature = negative_importance_features.iloc[-1].name
            print(f"Removing feature {worst_feature} with negative importance")
            columns.remove(worst_feature)
        else:
            break

    print(f"Final features: {columns}")
    return columns


column_names = [
    "SLOPE",
    "CAVES",
    "FAULTS",
    "WATERCO",
    "KARST_RO",
    "V_DOLINE",
    "AREA",
]
y_columns = {
    "BFI": "Baseflow index",
    "CV": "Coefficient of variation",
    "I": "Infiltration delay",
    "KA": "KA",
    "MAX": "Max",
    "ME": "Memory effect",
    "MEAN": "Mean",
    "MEDIAN": "Median",
    "MIN": "Min",
    "MIN-MAX": "Min-max",
    "RT": "Regulation time",
    "SIGMA-RT": "σ250/σ (%)",
    "STD": "Standard deviation",
    "VT": "Total volume",
    "SVC": "Spring variability coefficient",
    "RC": "Recession coefficient",
    "K": "Regulating power",
    "Q25-Q50": "Q25/Q50",
    "DK": "DK",
    "DI": "DI",
    "VD": "Dynamic volume",
}
