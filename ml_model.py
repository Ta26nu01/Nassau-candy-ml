"""
03_ml_models.py
Nassau Candy Distributor — Machine Learning Pipeline
  1. Regression  : Predict shipping lead time
  2. Classification: Label routes as Efficient / Delayed
  3. Clustering   : Discover route performance patterns
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    silhouette_score,
)

from sklearn.linear_model     import Ridge
from sklearn.ensemble         import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.cluster          import KMeans, AgglomerativeClustering
from sklearn.decomposition    import PCA

warnings.filterwarnings("ignore")

BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "../data/nassau_clean.csv")
MODDIR  = os.path.join(BASE, "../models")
OUTDIR  = os.path.join(BASE, "../outputs/ml")
os.makedirs(MODDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

BRAND   = "#4A154B"
SEED    = 42

# ── Feature engineering for ML ───────────────────────────────────────────────
CATEGORICAL = ["Ship Mode", "Factory", "Region", "Division"]
NUMERIC     = ["Sales", "Units", "Gross Profit", "Cost",
                "Order Month", "Order Quarter", "Order DayOfWeek"]

def build_features(df: pd.DataFrame):
    le_dict = {}
    X = df[NUMERIC + CATEGORICAL].copy()
    for col in CATEGORICAL:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    return X.astype(float), le_dict


def load_data():
    
    df = pd.read_csv(DATA, parse_dates=["Order Date", "Ship Date"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. REGRESSION — Predict Lead Time
# ══════════════════════════════════════════════════════════════════════════════
def run_regression(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("  MODULE 1 — REGRESSION: Predict Shipping Lead Time")
    print("═"*60)

    X, le_dict = build_features(df)
    y = df["Lead Time"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        "Ridge":            Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
        "Random Forest":    RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        "Gradient Boosting":GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                       max_depth=4, random_state=SEED),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        mae  = mean_absolute_error(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        r2   = r2_score(y_te, pred)
        cv   = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
        results[name] = dict(MAE=mae, RMSE=rmse, R2=r2, CV_R2=cv)
        print(f"  {name:<22} MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  CV-R²={cv:.4f}")

    # pick best by CV R²
    best_name = max(results, key=lambda k: results[k]["CV_R2"])
    best_model = models[best_name]
    print(f"\n  🏆  Best model: {best_name}")

    # Feature importance (tree-based)
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_,
                        index=NUMERIC + CATEGORICAL).sort_values(ascending=False)
    else:
        imp = pd.Series(np.abs(best_model.named_steps["m"].coef_),
                        index=NUMERIC + CATEGORICAL).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    imp.head(12).plot.barh(ax=ax, color=BRAND, alpha=0.8, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {best_name} (Lead Time Regression)",
                 fontweight="bold", color=BRAND)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "reg_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Actual vs predicted
    pred_best = best_model.predict(X_te)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_te, pred_best, alpha=0.35, s=10, color="#3498db", edgecolors="none")
    lims = [min(y_te.min(), pred_best.min()), max(y_te.max(), pred_best.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Lead Time (days)")
    ax.set_ylabel("Predicted Lead Time (days)")
    ax.set_title(f"Actual vs Predicted — {best_name}", fontweight="bold", color=BRAND)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "reg_actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump({"model": best_model, "le_dict": le_dict, "features": NUMERIC+CATEGORICAL,
                 "model_name": best_name, "results": results},
                os.path.join(MODDIR, "regression_model.pkl"))
    print(f"  💾  Saved → models/regression_model.pkl")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASSIFICATION — Efficient vs Delayed
# ══════════════════════════════════════════════════════════════════════════════
def run_classification(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("  MODULE 2 — CLASSIFICATION: Efficient vs Delayed")
    print("═"*60)

    X, le_dict = build_features(df)
    y = df["Is Delayed"].values
    print(f"  Class balance — Efficient: {(y==0).sum():,}  |  Delayed: {(y==1).sum():,}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               stratify=y, random_state=SEED)

    models = {
        "Random Forest":     RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                    random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                        max_depth=4, random_state=SEED),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        cv   = cross_val_score(model, X, y, cv=5, scoring="f1").mean()
        print(f"\n  ── {name} ──")
        print(classification_report(y_te, pred, target_names=["Efficient","Delayed"]))
        results[name] = {"model": model, "pred": pred, "CV_F1": cv}

    best_name = max(results, key=lambda k: results[k]["CV_F1"])
    best = results[best_name]
    print(f"  🏆  Best classifier: {best_name}  (CV F1={best['CV_F1']:.4f})")

    # Confusion matrix
    cm   = confusion_matrix(y_te, best["pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Efficient","Delayed"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold", color=BRAND)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "clf_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance
    imp = pd.Series(best["model"].feature_importances_,
                    index=NUMERIC + CATEGORICAL).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    imp.head(12).plot.barh(ax=ax, color="#e74c3c", alpha=0.8, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {best_name} (Delay Classification)",
                 fontweight="bold", color=BRAND)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "clf_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump({"model": best["model"], "le_dict": le_dict,
                 "features": NUMERIC+CATEGORICAL, "model_name": best_name},
                os.path.join(MODDIR, "classification_model.pkl"))
    print(f"  💾  Saved → models/classification_model.pkl")

    return {k: v["CV_F1"] for k, v in results.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLUSTERING — Route Performance Patterns
# ══════════════════════════════════════════════════════════════════════════════
def run_clustering(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("  MODULE 3 — CLUSTERING: Route Performance Patterns")
    print("═"*60)

    # Aggregate at route level
    route_df = df.groupby("Route").agg(
        Avg_Lead_Time  = ("Lead Time",       "mean"),
        Std_Lead_Time  = ("Lead Time",       "std"),
        Total_Orders   = ("Row ID",          "count"),
        Delay_Rate     = ("Is Delayed",      "mean"),
        Avg_Sales      = ("Sales",           "mean"),
        Avg_Profit     = ("Gross Profit",    "mean"),
        Avg_Efficiency = ("Efficiency Score","mean"),
    ).fillna(0).reset_index()

    features = ["Avg_Lead_Time","Std_Lead_Time","Total_Orders",
                "Delay_Rate","Avg_Sales","Avg_Profit","Avg_Efficiency"]
    sc   = StandardScaler()
    X_sc = sc.fit_transform(route_df[features])

    # Elbow + silhouette
    ks    = range(2, 8)
    inert = []
    silh  = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        lbl = km.fit_predict(X_sc)
        inert.append(km.inertia_)
        silh.append(silhouette_score(X_sc, lbl))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ks, inert, "o-", color=BRAND, lw=2)
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Curve", fontweight="bold")
    ax2.plot(ks, silh, "s-", color="#e74c3c", lw=2)
    ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette Score"); ax2.set_title("Silhouette Scores", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cluster_elbow_silhouette.png"), dpi=150, bbox_inches="tight")
    plt.close()

    best_k = ks[int(np.argmax(silh))]
    print(f"  Optimal k = {best_k}  (silhouette = {max(silh):.4f})")

    km_best = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
    route_df["Cluster"] = km_best.fit_predict(X_sc)

    # PCA 2D projection
    pca  = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_sc)
    route_df["PC1"] = X_2d[:, 0]
    route_df["PC2"] = X_2d[:, 1]

    CLUSTER_COLORS = ["#2ecc71","#e74c3c","#3498db","#f39c12","#9b59b6","#1abc9c"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in range(best_k):
        sub = route_df[route_df["Cluster"] == c]
        ax.scatter(sub["PC1"], sub["PC2"],
                   c=CLUSTER_COLORS[c], s=80, alpha=0.8,
                   edgecolors="white", lw=0.5, label=f"Cluster {c}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"Route Clusters (k={best_k}) — PCA 2D Projection",
                 fontsize=13, fontweight="bold", color=BRAND)
    ax.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cluster_pca_2d.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Cluster profile
    profile = route_df.groupby("Cluster")[features].mean().round(3)
    print("\n  Cluster profiles:")
    print(profile.to_string())

    # Assign human-readable labels
    label_map = {}
    for c in range(best_k):
        row = profile.loc[c]
        if row["Avg_Lead_Time"] == profile["Avg_Lead_Time"].min():
            label_map[c] = "⚡ Fast & Efficient"
        elif row["Delay_Rate"] == profile["Delay_Rate"].max():
            label_map[c] = "🔴 High Delay Risk"
        elif row["Total_Orders"] == profile["Total_Orders"].max():
            label_map[c] = "📦 High Volume"
        else:
            label_map[c] = f"🔵 Cluster {c}"
    route_df["Cluster Label"] = route_df["Cluster"].map(label_map)
    print("\n  Cluster labels:", label_map)

    # Radar / bar chart of cluster profiles
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(features))
    width = 0.8 / best_k
    norm_profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    for i, c in enumerate(range(best_k)):
        ax.bar(x + i*width, norm_profile.loc[c], width,
               label=label_map[c], color=CLUSTER_COLORS[i], alpha=0.8, edgecolor="white")
    ax.set_xticks(x + width*(best_k-1)/2)
    ax.set_xticklabels([f.replace("_"," ") for f in features], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Normalised Value")
    ax.set_title("Cluster Feature Profiles (Normalised)", fontsize=13, fontweight="bold", color=BRAND)
    ax.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cluster_profiles.png"), dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump({"model": km_best, "scaler": sc, "pca": pca,
                 "route_df": route_df, "features": features,
                 "label_map": label_map, "best_k": best_k},
                os.path.join(MODDIR, "clustering_model.pkl"))
    print(f"  💾  Saved → models/clustering_model.pkl")

    return route_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()
    print(f"  Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    reg_results   = run_regression(df)
    clf_results   = run_classification(df)
    cluster_df    = run_clustering(df)

    print("\n" + "="*60)
    print("  ✅  ALL MODELS TRAINED & SAVED")
    print("="*60)
    print("\n  Regression Results:")
    for name, r in reg_results.items():
        print(f"    {name:<22} R²={r['R2']:.4f}  CV-R²={r['CV_R2']:.4f}")
    print("\n  Classification CV-F1 Scores:")
    for name, f1 in clf_results.items():
        print(f"    {name:<22} CV-F1={f1:.4f}")
    print(f"\n  Clustering: see outputs/ml/cluster_*.png")
    print(f"\n  📁  Charts → {OUTDIR}/")
    print(f"  📁  Models → {MODDIR}/")
