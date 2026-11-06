"""Part A - clean the raw export, write the modelling table and the first delay evidence."""
import os
import pandas as pd
from src.shipment_prep import FEATURES, clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# --- "before" counts: after removing duplicates, before any filling ---
dedup = raw.drop_duplicates()
declared_text_missing = dedup["declared_value"].isna().sum()

clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
table.to_csv("data/modelling_table.csv", index=False)

# --- checks ---
q1, q3 = table["delivery_hours"].quantile(0.25), table["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
checked_cols = FEATURES + ["delivery_hours", "is_delayed"]

metrics = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
    "missing_parcel_weight_before": int(dedup["parcel_weight_kg"].isna().sum()),
    "sentinel_weight_rows": int((dedup["parcel_weight_kg"] > 100).sum()),
    "missing_declared_value_before": int(declared_text_missing),
    "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
    "service_type_spellings_before": int(dedup["service_type"].nunique()),
    "service_type_spellings_after": int(table["service_type"].nunique()),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(table),
    "nulls_in_features_and_target": int(table[checked_cols].isna().sum().sum()),
    "delayed_count": int(table["is_delayed"].sum()),
    "delay_rate": table["is_delayed"].mean(),
    "hours_q1": q1, "hours_q3": q3,
    "hours_lower_bound": lower, "hours_upper_bound": upper,
    "hours_outlier_count": int(((table["delivery_hours"] < lower) | (table["delivery_hours"] > upper)).sum()),
    "declared_value_mean_inr": table["declared_value_inr"].mean(),
    "declared_value_median_inr": table["declared_value_inr"].median(),
    "log_declared_value_mean": table["log_declared_value"].mean(),
    "log_declared_value_median": table["log_declared_value"].median(),
}
checks = pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())})
checks["value"] = checks["value"].astype(float).round(4)
checks.to_csv("outputs/part_a_checks.csv", index=False)

# --- delay evidence by group ---
table["distance_band"] = pd.cut(table["distance_km"], bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"], right=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = table.groupby(dim, observed=True).agg(shipments=("is_delayed", "size"),
                                              delayed=("is_delayed", "sum"),
                                              delay_rate=("is_delayed", "mean"),
                                              mean_hours=("delivery_hours", "mean")).reset_index()
    g.insert(0, "dimension", dim)
    g = g.rename(columns={dim: "group"})
    g["group"] = g["group"].astype(str)
    rows.append(g)
by_group = pd.concat(rows, ignore_index=True)
by_group[["delay_rate", "mean_hours"]] = by_group[["delay_rate", "mean_hours"]].round(4)
by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# --- correlations ---
corr = table.drop(columns=["distance_band"]).select_dtypes("number").corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv")

print(checks.to_string(index=False))
print(by_group.to_string(index=False))
top3 = corr["delivery_hours"].drop("delivery_hours").abs().sort_values(ascending=False).head(3).index
print({c: corr.loc[c, "delivery_hours"] for c in top3})
