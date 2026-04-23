"""
01_preprocessing.py
Nassau Candy Distributor — Data Cleaning & Feature Engineering
"""

import pandas as pd
import numpy as np
import os

# ── Factory metadata ───────────────────────────────────────────────────────────
FACTORY_COORDS = {
    "Lot's O' Nuts":     {"lat": 32.881893, "lon": -111.768036, "state": "Arizona"},
    "Wicked Choccy's":  {"lat": 32.076176, "lon": -81.088371,  "state": "Georgia"},
    "Sugar Shack":       {"lat": 48.11914,  "lon": -96.18115,   "state": "Minnesota"},
    "Secret Factory":    {"lat": 41.446333, "lon": -90.565487,  "state": "Illinois"},
    "The Other Factory": {"lat": 35.1175,   "lon": -89.971107,  "state": "Tennessee"},
}

PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":    "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":            "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":       "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":           "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":    "Wicked Choccy's",
    "Laffy Taffy":                          "Sugar Shack",
    "SweeTARTS":                            "Sugar Shack",
    "Nerds":                                "Sugar Shack",
    "Fun Dip":                              "Sugar Shack",
    "Fizzy Lifting Drinks":                 "Sugar Shack",
    "Everlasting Gobstopper":               "Secret Factory",
    "Lickable Wallpaper":                   "Secret Factory",
    "Wonka Gum":                            "Secret Factory",
    "Hair Toffee":                          "The Other Factory",
    "Kazookles":                            "The Other Factory",
}

DELAY_THRESHOLD = {
    "Same Day":      175,
    "First Class":   177,
    "Second Class":  179,
    "Standard Class": 181,
}


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── Parse dates ───────────────────────────────────────────────────────────
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True)

    # ── Lead time (normalised from encoded dates) ─────────────────────────────
    raw = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Lead Time"] = raw % 365          # strips the year-offset encoding

    # ── Drop invalid rows ─────────────────────────────────────────────────────
    df = df[df["Lead Time"] > 0].copy()

    # ── Factory assignment ────────────────────────────────────────────────────
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)
    df["Factory Lat"] = df["Factory"].map(lambda f: FACTORY_COORDS[f]["lat"])
    df["Factory Lon"] = df["Factory"].map(lambda f: FACTORY_COORDS[f]["lon"])
    df["Factory State"] = df["Factory"].map(lambda f: FACTORY_COORDS[f]["state"])

    # ── Route identifier ──────────────────────────────────────────────────────
    df["Route"] = df["Factory"] + " → " + df["State/Province"]
    df["Route Region"] = df["Factory"] + " → " + df["Region"]

    # ── Temporal features ─────────────────────────────────────────────────────
    df["Order Month"]     = df["Order Date"].dt.month
    df["Order Quarter"]   = df["Order Date"].dt.quarter
    df["Order DayOfWeek"] = df["Order Date"].dt.dayofweek

    # ── Delay flag (per ship-mode threshold) ──────────────────────────────────
    df["Delay Threshold"] = df["Ship Mode"].map(DELAY_THRESHOLD)
    df["Is Delayed"]      = (df["Lead Time"] > df["Delay Threshold"]).astype(int)

    # ── Efficiency score (inverted lead-time, normalised 0-100) ──────────────
    lt_min, lt_max = df["Lead Time"].min(), df["Lead Time"].max()
    df["Efficiency Score"] = 100 * (lt_max - df["Lead Time"]) / (lt_max - lt_min)

    # ── Profit margin ─────────────────────────────────────────────────────────
    df["Profit Margin %"] = (df["Gross Profit"] / df["Sales"] * 100).round(2)

    return df


def save_clean(df: pd.DataFrame, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "nassau_clean.csv")
    df.to_csv(out, index=False)
    print(f"✅  Saved cleaned dataset → {out}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    return out


if __name__ == "__main__":
    RAW  = os.path.join(os.path.dirname(__file__), "Nassau_Candy_Distributor.csv")
    OUT  = os.path.dirname(__file__)
    df   = load_and_clean(RAW)
    save_clean(df, OUT)

    print("\n── Sample ──")
    print(df[["Route", "Ship Mode", "Lead Time", "Is Delayed", "Efficiency Score"]].head(8).to_string(index=False))
    print("\n── Lead Time by Ship Mode ──")
    print(df.groupby("Ship Mode")["Lead Time"].agg(["mean","median","std","min","max"]).round(2))
