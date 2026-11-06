"""Part C - delay classifier, feature signal, tuning and the alert threshold."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]

# 1. stratified split, same two-step recipe as Part B
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "delayed", y_train.sum(), y_val.sum(), y_test.sum())


def cm_row(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0)}


def write(df, path):
    num = df.select_dtypes("float").columns
    df[num] = df[num].round(4)
    df.to_csv(path, index=False)


# 2. blind baseline + three classifiers (default 0.5 rule)
rows = [{"model": "blind_all_on_time", **cm_row(y_val, np.zeros(len(y_val), dtype=int))}]

logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
rows.append({"model": "logistic_regression", **cm_row(y_val, logit.predict(X_val))})

best_d, best_f1 = None, -1
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f = f1_score(y_val, t.predict(X_val))
    if f > best_f1:                       # strict > keeps the smaller depth on ties
        best_d, best_f1 = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
rows.append({"model": f"decision_tree_depth_{best_d}", **cm_row(y_val, tree.predict(X_val))})

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rows.append({"model": "random_forest_default", **cm_row(y_val, rf.predict(X_val))})
write(pd.DataFrame(rows), "outputs/part_c_models.csv")

# 3. which columns carry signal?
skb_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
scores = pd.DataFrame({"feature": FEATURES, "f_score": skb_all.scores_, "rf_importance": rf.feature_importances_})
scores["f_rank"] = scores["f_score"].rank(ascending=False, method="first").astype(int)
scores["is_noise"] = scores["feature"].isin(NOISE_COLUMNS)
scores = scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
write(scores, "outputs/part_c_feature_scores.csv")

k_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("sel", SelectKBest(f_classif, k=k)),
                  ("rf", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    k_rows.append({"k": k, "val_f1": f1_score(y_val, p.predict(X_val))})
write(pd.DataFrame(k_rows), "outputs/part_c_selectk.csv")

# 4. tuning on 7 selected features (selector fitted on training rows only)
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)
Xtr7, Xva7 = selector.transform(X_train), selector.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}

gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Xtr7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)

tuning = pd.DataFrame([
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 27 * 5,
     "best_params": json.dumps(gs.best_params_, sort_keys=True), "best_cv_accuracy": gs.best_score_,
     "val_f1": f1_score(y_val, gs.best_estimator_.predict(Xva7))},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 9 * 5,
     "best_params": json.dumps(rs.best_params_, sort_keys=True), "best_cv_accuracy": rs.best_score_,
     "val_f1": f1_score(y_val, rs.best_estimator_.predict(Xva7))},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": json.dumps({"max_depth": "unlimited", "min_samples_leaf": 1, "n_estimators": 100}, sort_keys=True),
     "best_cv_accuracy": np.nan, "val_f1": f1_score(y_val, default7.predict(Xva7))},
])
write(tuning, "outputs/part_c_tuning.csv")

# 5. final configuration: highest val_f1, ties -> default, grid, randomized
order = ["default_forest_same_7_features", "grid_search", "randomized_search"]
t_idx = tuning.set_index("search").loc[order]
chosen_from = t_idx["val_f1"].round(10).idxmax()           # idxmax returns the first max in `order`
if chosen_from == "grid_search":
    params = gs.best_params_
elif chosen_from == "randomized_search":
    params = rs.best_params_
else:
    params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}

final = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                  ("model", RandomForestClassifier(random_state=42, **params))]).fit(X_train, y_train)

# 6. alert threshold on validation probabilities
p_val = final.predict_proba(X_val)[:, 1]
th_rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    th_rows.append({"threshold": t, **cm_row(y_val, (p_val >= t).astype(int))})
th = pd.DataFrame(th_rows)
ok = th[th["recall"] >= 0.80]
threshold = float(ok["threshold"].max()) if len(ok) else 0.1
write(th, "outputs/part_c_thresholds.csv")

# 7. one test evaluation at the chosen threshold
p_test = final.predict_proba(X_test)[:, 1]
write(pd.DataFrame([{"threshold": threshold, **cm_row(y_test, (p_test >= threshold).astype(int))}]),
      "outputs/part_c_final_test.csv")
write(pd.DataFrame([{"selector_k": K, "n_estimators": params["n_estimators"],
                     "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
                     "min_samples_leaf": params["min_samples_leaf"], "threshold": threshold,
                     "chosen_from": chosen_from}]), "outputs/part_c_final_config.csv")

for f in ["models", "feature_scores", "selectk", "tuning", "thresholds", "final_test", "final_config"]:
    print(f); print(pd.read_csv(f"outputs/part_c_{f}.csv").to_string(index=False))
print("noise importance share", scores.loc[scores.is_noise, "rf_importance"].sum())
