"""
02_eda.py
Nassau Candy Distributor — Exploratory Data Analysis
Generates charts saved to outputs/eda/
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "../data/nassau_clean.csv")
OUTDIR = os.path.join(BASE, "../outputs/eda")
os.makedirs(OUTDIR, exist_ok=True)

PALETTE = {
    "Same Day":      "#2ecc71",
    "First Class":   "#3498db",
    "Second Class":  "#f39c12",
    "Standard Class":"#e74c3c",
}
BRAND = "#4A154B"   # deep purple accent


def load() -> pd.DataFrame:
    if not os.path.exists(DATA):
        sys.path.insert(0, BASE)
        from preprocessing_01 import load_and_clean, save_clean
        df = load_and_clean(os.path.join(BASE, "../data/Nassau_Candy_Distributor.csv"))
        save_clean(df, os.path.join(BASE, "../data"))
    return pd.read_csv(DATA, parse_dates=["Order Date","Ship Date"])


def fig_save(name):
    path = os.path.join(OUTDIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾  {name}")


# ── 1. Lead-time distribution by ship mode ────────────────────────────────────
def plot_lead_time_dist(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    modes = ["Same Day","First Class","Second Class","Standard Class"]
    for ax, mode in zip(axes, modes):
        sub = df[df["Ship Mode"] == mode]["Lead Time"]
        ax.hist(sub, bins=20, color=PALETTE[mode], edgecolor="white", alpha=0.85)
        ax.axvline(sub.mean(), color="black", lw=1.5, ls="--", label=f"Mean: {sub.mean():.1f}d")
        ax.set_title(mode, fontsize=11, fontweight="bold")
        ax.set_xlabel("Lead Time (days)")
        ax.set_ylabel("Orders")
        ax.legend(fontsize=9)
    fig.suptitle("Lead Time Distribution by Ship Mode", fontsize=14, fontweight="bold", color=BRAND)
    plt.tight_layout()
    fig_save("01_lead_time_dist.png")


# ── 2. Average lead time by region ───────────────────────────────────────────
def plot_region_lead(df):
    grp = df.groupby(["Region","Ship Mode"])["Lead Time"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    regions = grp["Region"].unique()
    x = np.arange(len(regions))
    width = 0.2
    for i, mode in enumerate(["Same Day","First Class","Second Class","Standard Class"]):
        sub = grp[grp["Ship Mode"] == mode].set_index("Region").reindex(regions)
        ax.bar(x + i*width, sub["Lead Time"], width, label=mode, color=PALETTE[mode], alpha=0.85)
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(regions, fontsize=10)
    ax.set_ylabel("Avg Lead Time (days)")
    ax.set_title("Average Lead Time by Region & Ship Mode", fontsize=13, fontweight="bold", color=BRAND)
    ax.legend(title="Ship Mode", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig_save("02_region_lead_time.png")


# ── 3. Top 10 most & least efficient routes ───────────────────────────────────
def plot_route_efficiency(df):
    route_grp = (df.groupby("Route")
                   .agg(Avg_Lead=("Lead Time","mean"),
                        Orders=("Row ID","count"),
                        Delay_Rate=("Is Delayed","mean"))
                   .reset_index()
                   .sort_values("Avg_Lead"))
    top10  = route_grp.head(10)
    bot10  = route_grp.tail(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, color in [
        (ax1, top10, "Top 10 Most Efficient Routes", "#2ecc71"),
        (ax2, bot10, "Top 10 Least Efficient Routes", "#e74c3c"),
    ]:
        bars = ax.barh(data["Route"], data["Avg_Lead"], color=color, alpha=0.85, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f d", padding=3, fontsize=8)
        ax.set_xlabel("Avg Lead Time (days)")
        ax.set_title(title, fontsize=11, fontweight="bold", color=BRAND)
        ax.invert_yaxis()
    plt.tight_layout()
    fig_save("03_route_efficiency.png")


# ── 4. Delay rate heatmap (Factory × Region) ─────────────────────────────────
def plot_delay_heatmap(df):
    pivot = df.pivot_table(index="Factory", columns="Region",
                           values="Is Delayed", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                linewidths=0.5, ax=ax, cbar_kws={"label":"Delay Rate (%)"})
    ax.set_title("Delay Rate (%) by Factory → Region", fontsize=13, fontweight="bold", color=BRAND)
    plt.tight_layout()
    fig_save("04_delay_heatmap.png")


# ── 5. Monthly shipping volume trend ─────────────────────────────────────────
def plot_monthly_trend(df):
    df["YM"] = df["Order Date"].dt.to_period("M").astype(str)
    monthly = df.groupby(["YM","Ship Mode"]).size().reset_index(name="Orders")
    fig, ax = plt.subplots(figsize=(13, 5))
    for mode in ["Same Day","First Class","Second Class","Standard Class"]:
        sub = monthly[monthly["Ship Mode"] == mode]
        ax.plot(sub["YM"], sub["Orders"], label=mode, color=PALETTE[mode], lw=1.5, marker="o", ms=3)
    ax.set_xlabel("Month")
    ax.set_ylabel("Orders")
    ax.set_title("Monthly Order Volume by Ship Mode", fontsize=13, fontweight="bold", color=BRAND)
    ax.legend(title="Ship Mode")
    step = max(1, len(monthly["YM"].unique()) // 12)
    ticks = monthly["YM"].unique()[::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    fig_save("05_monthly_trend.png")


# ── 6. Division profitability ─────────────────────────────────────────────────
def plot_division_profit(df):
    grp = df.groupby("Division").agg(
        Total_Sales=("Sales","sum"),
        Total_Profit=("Gross Profit","sum"),
        Orders=("Row ID","count")
    ).reset_index()
    grp["Margin%"] = grp["Total_Profit"] / grp["Total_Sales"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#9b59b6","#3498db","#f39c12"]
    ax1.bar(grp["Division"], grp["Total_Sales"], color=colors, alpha=0.85, edgecolor="white")
    ax1.set_ylabel("Total Sales ($)")
    ax1.set_title("Total Sales by Division", fontweight="bold", color=BRAND)
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:,.0f}"))

    ax2.bar(grp["Division"], grp["Margin%"], color=colors, alpha=0.85, edgecolor="white")
    ax2.set_ylabel("Gross Margin (%)")
    ax2.set_title("Profit Margin by Division", fontweight="bold", color=BRAND)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    fig_save("06_division_profitability.png")


# ── 7. Efficiency score boxplot by factory ────────────────────────────────────
def plot_factory_efficiency(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    factories = df["Factory"].unique()
    data = [df[df["Factory"] == f]["Efficiency Score"].values for f in factories]
    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops=dict(color="black", lw=2))
    colors = ["#9b59b6","#e74c3c","#3498db","#2ecc71","#f39c12"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels([f.replace(" ", "\n") for f in factories], fontsize=9)
    ax.set_ylabel("Efficiency Score (0–100)")
    ax.set_title("Shipping Efficiency Score by Factory", fontsize=13, fontweight="bold", color=BRAND)
    plt.tight_layout()
    fig_save("07_factory_efficiency.png")


# ── 8. Correlation heatmap ────────────────────────────────────────────────────
def plot_correlation(df):
    num_cols = ["Lead Time","Sales","Units","Gross Profit","Cost",
                "Profit Margin %","Efficiency Score","Is Delayed",
                "Order Month","Order Quarter"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", color=BRAND)
    plt.tight_layout()
    fig_save("08_correlation_heatmap.png")


def main():
    print("📊  Running EDA …")
    df = load()
    plot_lead_time_dist(df)
    plot_region_lead(df)
    plot_route_efficiency(df)
    plot_delay_heatmap(df)
    plot_monthly_trend(df)
    plot_division_profit(df)
    plot_factory_efficiency(df)
    plot_correlation(df)
    print(f"\n✅  All EDA charts saved to  {OUTDIR}/")


if __name__ == "__main__":
    main()