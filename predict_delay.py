"""Part D (3/3) - turn new parcels into an action list (uses the shared cleaning/feature code)."""
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

new_raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute["column"], impute["median"]))
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new_raw, medians=medians)
feats = build_features(clean, hubs)
X = feats[feature_order]

out = feats[["shipment_id", "hub_id"]].copy()
out = out.merge(segments, on="hub_id", how="left")
out["delay_probability"] = model.predict_proba(X)[:, 1].round(4)
out["flagged"] = (out["delay_probability"] >= threshold).astype(int)
out["action"] = "No action"
out.loc[out["flagged"] == 1, "action"] = "Proactive customer call"
out.loc[(out["flagged"] == 1) & (out["segment"] == "Escalate"), "action"] = "Escalate to hub manager"
out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)

print(out["action"].value_counts().to_string())
print("rows", len(out), "flagged", int(out["flagged"].sum()))
