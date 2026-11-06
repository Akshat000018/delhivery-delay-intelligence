"""Part D (1/3) - hub profile, K-Means segments, separation evidence and tier check."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# 1. hub profile, sorted by ascending hub_id (groupby order) - this order is fixed for clustering
profile = (table.groupby("hub_id")
           .agg(shipments=("shipment_id", "size"),
                mean_delivery_hours=("delivery_hours", "mean"),
                delay_rate=("is_delayed", "mean"),
                mean_distance_km=("distance_km", "mean"),
                mean_n_stops=("n_stops", "mean"))
           .reset_index().sort_values("hub_id").reset_index(drop=True))
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# 2. scale the four behaviour columns and cluster
cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(profile[cols])


def across_over_within(Z, labels):
    """Mean distance of between-cluster hub pairs / mean distance of within-cluster pairs (276 pairs)."""
    i, j = np.triu_indices(len(Z), k=1)
    d = np.linalg.norm(Z[i] - Z[j], axis=1)
    same = labels[i] == labels[j]
    return d[~same].mean() / d[same].mean()


# 3. separation evidence for K = 2..5
sep_rows = []
for k in [2, 3, 4, 5]:
    lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
    sizes = sorted(np.bincount(lab), reverse=True)
    sep_rows.append({"K": k, "across_over_within": across_over_within(Z, lab),
                     "cluster_sizes": "-".join(str(s) for s in sizes)})
pd.DataFrame(sep_rows).round({"across_over_within": 4}).to_csv("outputs/part_d_separation.csv", index=False)

# 4. final K = 3 (fixed by the business); name clusters by mean delay rate, not by cluster id
labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(Z)
profile["cluster"] = labels
rank = profile.groupby("cluster")["delay_rate"].mean().sort_values().index.tolist()
names = dict(zip(rank, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
segments.to_csv("outputs/hub_segments.csv", index=False)

seg_profile = (profile.groupby("segment")
               .agg(hubs=("hub_id", "size"), mean_delivery_hours=("mean_delivery_hours", "mean"),
                    delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
                    mean_n_stops=("mean_n_stops", "mean"))
               .reindex(["Standard", "Watch", "Escalate"]).reset_index())
seg_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

tier_implies = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = segments[segments["segment"] != segments["hub_tier"].map(tier_implies)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

for f in ["hub_profile", "part_d_separation", "segment_profile", "tier_vs_segment_mismatch"]:
    print(f); print(pd.read_csv(f"outputs/{f}.csv").to_string(index=False))
print(segments.to_string(index=False))
