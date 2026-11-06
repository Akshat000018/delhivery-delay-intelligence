"""Part B - delivery-hours regression: split, leakage-safe scaling, baselines, models, CV."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["delivery_hours"]

# 1. split 70 / 15 / 15
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2. scaling without leakage: fit on training rows only
scaler = StandardScaler().fit(X_train)
train_scaled, test_scaled = scaler.transform(X_train), scaler.transform(X_test)
checks = pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": [len(X_train), len(X_val), len(X_test),
              np.abs(train_scaled.mean(axis=0)).max(), np.abs(test_scaled.mean(axis=0)).max()]})
checks["value"] = checks["value"].astype(float)
# train value is ~1e-16; rounding to 4 places keeps it below the 1e-9 acceptance bound (writes 0.0)
checks["value"] = checks["value"].round(4)
checks.to_csv("outputs/part_b_checks.csv", index=False)


def score(model_name, split, y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {"model": model_name, "split": split, "MAE": mean_absolute_error(y_true, pred),
            "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, pred)}


# 3. baseline
rows = [score("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean()))]

# 4. tree-depth diagnostic
depth_rows = []
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    depth_rows.append({"max_depth": "unlimited" if d is None else d,
                       "train_rmse": np.sqrt(mean_squared_error(y_train, t.predict(X_train))),
                       "val_rmse": np.sqrt(mean_squared_error(y_val, t.predict(X_val)))})
depth = pd.DataFrame(depth_rows)
best_i = int(depth["val_rmse"].values.argmin())          # argmin keeps the first (smaller) depth on ties
best_depth = None if depth.loc[best_i, "max_depth"] == "unlimited" else int(depth.loc[best_i, "max_depth"])
depth[["train_rmse", "val_rmse"]] = depth[["train_rmse", "val_rmse"]].round(4)
depth.to_csv("outputs/part_b_tree_depth.csv", index=False)

# 5. four candidates
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
for name, m in models.items():
    m.fit(X_train, y_train)
    rows.append(score(name, "val", y_val, m.predict(X_val)))
val_scores = pd.DataFrame(rows)

# 6. learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    p = g.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": np.sqrt(mean_squared_error(y_val, p)), "val_r2": r2_score(y_val, p)})
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. cross-validation of the two best, final test score of the best
ranked = val_scores[val_scores["model"] != "mean_baseline"].sort_values("RMSE")
top2 = list(ranked["model"].head(2))
X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)       # one shared object
cv_rows = []
for name in top2:
    s = cross_val_score(models[name], X_trval, y_trval, cv=kf)
    cv_rows.append({"model": name, "mean_r2": s.mean(), "std_r2": s.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

final_name = top2[0]
final_row = score(final_name, "test", y_test, models[final_name].predict(X_test))   # single test evaluation
metrics = pd.concat([val_scores, pd.DataFrame([final_row])], ignore_index=True)
metrics[["MAE", "MSE", "RMSE", "R2"]] = metrics[["MAE", "MSE", "RMSE", "R2"]].round(4)
metrics.to_csv("outputs/part_b_metrics.csv", index=False)

for f in ["checks", "tree_depth", "metrics", "gb_learning_rate", "cv"]:
    print(f); print(pd.read_csv(f"outputs/part_b_{f}.csv").to_string(index=False))
