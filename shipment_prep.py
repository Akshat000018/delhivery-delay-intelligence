"""Shared constants and preparation functions for the Delay Intelligence pipeline.

Used by Part A (training table), Parts B/C/D (modelling) and predict_delay.py (scoring),
so that training and prediction go through exactly the same cleaning and feature code.
"""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
WEIGHT_SENTINEL_LIMIT = 100  # network accepts parcels up to 40 kg; anything above 100 is a scanner fault


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export. Returns (clean_df, medians).

    Steps, in order: drop duplicates -> normalise service_type -> flag unknown payment mode ->
    parse declared value -> parse dispatch time -> sentinel weights to missing -> median fill.
    If `medians` is None they are computed from the non-missing values after steps 1-6;
    otherwise the dictionary passed in is used (as when scoring new parcels).
    """
    df = raw.drop_duplicates().copy()

    df["service_type"] = df["service_type"].str.strip().str.title()
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    value_text = (df["declared_value"].astype("string")
                  .str.replace("\u20b9", "", regex=False)
                  .str.replace(",", "", regex=False)
                  .str.strip())
    df["declared_value_inr"] = pd.to_numeric(value_text, errors="coerce").astype("float64")
    df = df.drop(columns=["declared_value"])

    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    df.loc[df["parcel_weight_kg"] > WEIGHT_SENTINEL_LIMIT, "parcel_weight_kg"] = np.nan

    if medians is None:
        medians = {c: float(df[c].median()) for c in MEDIAN_COLUMNS}
    else:
        medians = {c: float(medians[c]) for c in MEDIAN_COLUMNS}
    for c in MEDIAN_COLUMNS:
        df[c] = df[c].fillna(medians[c])

    return df.reset_index(drop=True), medians


def build_features(clean, hubs):
    """Attach hub columns and derive model features. Never touches delivery_hours."""
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    out = clean.merge(hubs[hub_cols], on="hub_id", how="left")
    out["promised_hours"] = out["service_type"].map(PROMISED_HOURS).astype(int)
    out["is_cod"] = (out["payment_mode"] == "COD").astype(int)
    out["is_weekend"] = (out["dispatch_time"].dt.dayofweek >= 5).astype(int)
    out["dispatch_hour"] = out["dispatch_time"].dt.hour.astype(int)
    out["log_declared_value"] = np.log(out["declared_value_inr"])
    return out
