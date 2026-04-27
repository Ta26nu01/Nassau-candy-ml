"""
dashboard.py
Nassau Candy Distributor — Streamlit Route Efficiency Dashboard
Run: streamlit run dashboard.py
"""
import os
import sys
import subprocess
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths (cloud-safe — everything relative to THIS file) ─────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "nassau_clean.csv")
RAW     = os.path.join(BASE, "Nassau_Candy_Distributor.csv")
REG_PKL = os.path.join(BASE, "models", "regression_model.pkl")
CLF_PKL = os.path.join(BASE, "models", "classification_model.pkl")
CLU_PKL = os.path.join(BASE, "models", "clustering_model.pkl")
EDA_OUT = os.path.join(BASE, "outputs", "ml")
PRE_SCRIPT = os.path.join(BASE, "preprocessing.py")
ML_SCRIPT  = os.path.join(BASE, "ml_model.py")

# ── Page config (must come before any other st. calls) ────────────────────────
st.set_page_config(
    page_title="Nassau Candy — Route Efficiency",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-setup: runs preprocessing + ML if outputs are missing ────────────────
def run_setup():
    os.makedirs(os.path.join(BASE, "models"),        exist_ok=True)
    os.makedirs(os.path.join(BASE, "outputs", "ml"), exist_ok=True)
    if not os.path.exists(RAW):
        st.error(
            "❌ Raw data file not found!\n\n"
            f"Expected at: `{RAW}`\n\n"
            "Please make sure `Nassau_Candy_Distributor.csv` is in the repo root."
        )
        st.stop()
    if not os.path.exists(DATA):
        with st.spinner("⚙️ First launch — cleaning data (30–60 sec)..."):
            result = subprocess.run(
                [sys.executable, PRE_SCRIPT],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                st.error(f"❌ Preprocessing failed:\n```\n{result.stderr}\n```")
                st.stop()
    if not os.path.exists(REG_PKL):
        with st.spinner("🤖 Training ML models — only happens once (1–2 min)..."):
            result = subprocess.run(
                [sys.executable, ML_SCRIPT],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                st.error(f"❌ ML training failed:\n```\n{result.stderr}\n```")
                st.stop()

run_setup()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1a0a2e; }
[data-testid="stSidebar"] * { color: #e8d5f5 !important; }
.metric-card {
    background: linear-gradient(135deg,#2d1b4e,#1a0a2e);
    border-radius:12px; padding:18px 22px; margin:6px 0;
    border-left:4px solid #9b59b6;
}
.metric-val { font-size:2rem; font-weight:800; color:#e8d5f5; }
.metric-lbl { font-size:.85rem; color:#b39ddb; margin-top:2px; }
h1,h2,h3 { color: #4A154B !important; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { border-radius:8px 8px 0 0; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Factory coordinates for map ───────────────────────────────────────────────
FACTORY_COORDS = {
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":  (32.076176, -81.088371),
    "Sugar Shack":       (48.11914,  -96.18115),
    "Secret Factory":    (41.446333, -90.565487),
    "The Other Factory": (35.1175,   -89.971107),
}

US_STATE_COORDS = {
    "Alabama": (32.8, -86.8), "Alaska": (61.4, -152.0), "Arizona": (34.0, -111.1),
    "Arkansas": (34.8, -92.2), "California": (36.8, -119.4), "Colorado": (39.1, -105.5),
    "Connecticut": (41.6, -72.7), "Delaware": (39.0, -75.5), "Florida": (27.8, -81.5),
    "Georgia": (32.7, -83.4), "Hawaii": (20.9, -157.0), "Idaho": (44.2, -114.5),
    "Illinois": (40.0, -89.2), "Indiana": (40.3, -86.1), "Iowa": (42.1, -93.5),
    "Kansas": (38.5, -98.4), "Kentucky": (37.5, -85.3), "Louisiana": (31.2, -92.1),
    "Maine": (45.4, -69.0), "Maryland": (39.0, -76.8), "Massachusetts": (42.2, -71.5),
    "Michigan": (44.4, -85.4), "Minnesota": (46.4, -93.1), "Mississippi": (32.7, -89.7),
    "Missouri": (38.5, -92.5), "Montana": (47.0, -110.5), "Nebraska": (41.5, -99.9),
    "Nevada": (39.5, -116.9), "New Hampshire": (43.7, -71.6), "New Jersey": (40.1, -74.5),
    "New Mexico": (34.4, -106.1), "New York": (42.2, -74.9), "North Carolina": (35.6, -79.8),
    "North Dakota": (47.5, -100.5), "Ohio": (40.4, -82.8), "Oklahoma": (35.5, -97.5),
    "Oregon": (44.1, -120.5), "Pennsylvania": (40.9, -77.8), "Rhode Island": (41.6, -71.5),
    "South Carolina": (33.9, -80.9), "South Dakota": (44.4, -100.2), "Tennessee": (35.9, -86.7),
    "Texas": (31.5, -99.3), "Utah": (39.3, -111.1), "Vermont": (44.0, -72.7),
    "Virginia": (37.8, -78.2), "Washington": (47.4, -120.5), "West Virginia": (38.6, -80.5),
    "Wisconsin": (44.3, -89.8), "Wyoming": (43.0, -107.6), "District of Columbia": (38.9, -77.0),
}

PALETTE = {
    "Same Day":       "#2ecc71",
    "First Class":    "#3498db",
    "Second Class":   "#f39c12",
    "Standard Class": "#e74c3c",
}
BRAND = "#4A154B"

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA, parse_dates=["Order Date", "Ship Date"])
    return df

@st.cache_resource
def load_models():
    models = {}
    for key, path in [("reg", REG_PKL), ("clf", CLF_PKL), ("clu", CLU_PKL)]:
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models

df_raw = load_data()
models = load_models()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://em-content.zobj.net/source/apple/391/candy_1f36c.png", width=60)
    st.title("Nassau Candy 🍬")
    st.markdown("---")
    st.subheader("🔍 Filters")

    date_min   = df_raw["Order Date"].min().date()
    date_max   = df_raw["Order Date"].max().date()
    date_range = st.date_input(
        "Order Date Range", [date_min, date_max],
        min_value=date_min, max_value=date_max,
        key="sidebar_date_range"
    )
    regions    = st.multiselect("Region",    sorted(df_raw["Region"].unique()),
                                default=sorted(df_raw["Region"].unique()),
                                key="sidebar_regions")
    ship_modes = st.multiselect("Ship Mode", sorted(df_raw["Ship Mode"].unique()),
                                default=sorted(df_raw["Ship Mode"].unique()),
                                key="sidebar_ship_modes")
    factories  = st.multiselect("Factory",   sorted(df_raw["Factory"].unique()),
                                default=sorted(df_raw["Factory"].unique()),
                                key="sidebar_factories")
    lt_thresh  = st.slider(
        "Lead Time Threshold (days)",
        int(df_raw["Lead Time"].min()),
        int(df_raw["Lead Time"].max()),
        int(df_raw["Lead Time"].quantile(0.75)),
        key="sidebar_lt_thresh"
    )
    st.markdown("---")
    st.caption("Nassau Candy Distributor\nRoute Efficiency Analytics")

# ── Filter data ───────────────────────────────────────────────────────────────
df = df_raw.copy()
if len(date_range) == 2:
    df = df[
        (df["Order Date"].dt.date >= date_range[0]) &
        (df["Order Date"].dt.date <= date_range[1])
    ]
df = df[
    df["Region"].isin(regions) &
    df["Ship Mode"].isin(ship_modes) &
    df["Factory"].isin(factories)
]
df_above_thresh = df[df["Lead Time"] > lt_thresh]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🍬 Nassau Candy — Shipping Route Efficiency Dashboard")
st.markdown(
    f"**{df.shape[0]:,} orders** | {df['Route'].nunique()} routes | "
    f"{df['State/Province'].nunique()} states | "
    f"{df['Factory'].nunique()} factories"
)
st.markdown("---")

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, "📦 Total Orders",   f"{df.shape[0]:,}",                    ""),
    (c2, "⏱ Avg Lead Time",  f"{df['Lead Time'].mean():.1f}d",       "across all routes"),
    (c3, "🔴 Delay Rate",     f"{df['Is Delayed'].mean()*100:.1f}%", "above threshold"),
    (c4, "⚡ Avg Efficiency", f"{df['Efficiency Score'].mean():.1f}", "/ 100"),
    (c5, "💰 Total Revenue",  f"${df['Sales'].sum():,.0f}",           "gross sales"),
]
for col, lbl, val, sub in kpis:
    col.markdown(f"""<div class='metric-card'>
        <div class='metric-lbl'>{lbl}</div>
        <div class='metric-val'>{val}</div>
        <div class='metric-lbl'>{sub}</div>
    </div>""", unsafe_allow_html=True)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview", "🗺️ Geo Map", "🚚 Ship Mode",
    "🔬 Route Drill-Down", "🤖 ML Insights", "🔮 Predictor"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Route Performance Overview")
    col1, col2 = st.columns(2)

    with col1:
        route_grp = (
            df.groupby("Route")
              .agg(Avg_Lead=("Lead Time", "mean"),
                   Orders=("Row ID", "count"),
                   Delay_Rate=("Is Delayed", "mean"))
              .reset_index()
              .sort_values("Avg_Lead")
        )
        top10 = route_grp.head(10)
        fig = px.bar(top10, x="Avg_Lead", y="Route", orientation="h",
                     color="Avg_Lead", color_continuous_scale="Greens_r",
                     title="Top 10 Most Efficient Routes",
                     labels={"Avg_Lead": "Avg Lead Time (days)", "Route": "Route"})
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                          height=420, plot_bgcolor="rgba(0,0,0,0)",
                          title_font_color=BRAND)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        bot10 = route_grp.tail(10)
        fig = px.bar(bot10, x="Avg_Lead", y="Route", orientation="h",
                     color="Avg_Lead", color_continuous_scale="Reds",
                     title="Top 10 Least Efficient Routes",
                     labels={"Avg_Lead": "Avg Lead Time (days)", "Route": "Route"})
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                          height=420, plot_bgcolor="rgba(0,0,0,0)",
                          title_font_color=BRAND)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Lead Time Distribution")
    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(df, x="Lead Time", color="Ship Mode",
                           barmode="overlay", nbins=30,
                           color_discrete_map=PALETTE,
                           title="Lead Time Distribution by Ship Mode")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", title_font_color=BRAND)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(df, x="Factory", y="Lead Time", color="Ship Mode",
                     color_discrete_map=PALETTE,
                     title="Lead Time by Factory & Ship Mode")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", title_font_color=BRAND,
                          xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    df["YM"] = df["Order Date"].dt.to_period("M").astype(str)
    monthly  = df.groupby(["YM", "Ship Mode"]).size().reset_index(name="Orders")
    fig = px.line(monthly, x="YM", y="Orders", color="Ship Mode",
                  color_discrete_map=PALETTE,
                  title="Monthly Order Volume by Ship Mode",
                  markers=True)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", title_font_color=BRAND,
                      xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GEO MAP
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🗺️ Geographic Shipping Efficiency Map")

    map_metric = st.selectbox(
        "Map Metric",
        ["Avg Lead Time", "Delay Rate %", "Total Orders", "Avg Efficiency Score"],
        key="geo_map_metric"
    )

    state_grp = df.groupby("State/Province").agg(
        Avg_Lead       = ("Lead Time",        "mean"),
        Delay_Rate     = ("Is Delayed",       "mean"),
        Total_Orders   = ("Row ID",           "count"),
        Avg_Efficiency = ("Efficiency Score", "mean"),
    ).reset_index()
    state_grp["Delay_Rate"] *= 100

    col_map = {
        "Avg Lead Time":        ("Avg_Lead",       "RdYlGn_r", "Avg Lead Time (days)"),
        "Delay Rate %":         ("Delay_Rate",     "Reds",     "Delay Rate (%)"),
        "Total Orders":         ("Total_Orders",   "Blues",    "Total Orders"),
        "Avg Efficiency Score": ("Avg_Efficiency", "Greens",   "Efficiency Score"),
    }
    col_key, cscale, cbar_lbl = col_map[map_metric]

    state_grp["lat"] = state_grp["State/Province"].map(
        lambda s: US_STATE_COORDS.get(s, (None, None))[0])
    state_grp["lon"] = state_grp["State/Province"].map(
        lambda s: US_STATE_COORDS.get(s, (None, None))[1])
    state_grp = state_grp.dropna(subset=["lat", "lon"])

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=state_grp["lon"], lat=state_grp["lat"],
        text=state_grp.apply(lambda r:
            f"<b>{r['State/Province']}</b><br>"
            f"Avg Lead Time: {r['Avg_Lead']:.1f}d<br>"
            f"Delay Rate: {r['Delay_Rate']:.1f}%<br>"
            f"Orders: {r['Total_Orders']:,}<br>"
            f"Efficiency: {r['Avg_Efficiency']:.1f}", axis=1),
        hoverinfo="text",
        marker=dict(
            size=state_grp[col_key] / state_grp[col_key].max() * 30 + 8,
            color=state_grp[col_key],
            colorscale=cscale,
            showscale=True,
            colorbar_title=cbar_lbl,
            line_width=0.5, line_color="white",
            opacity=0.8,
        ),
        name="Customer States",
    ))
    for fname, (flat, flon) in FACTORY_COORDS.items():
        if fname in df["Factory"].values:
            fig.add_trace(go.Scattergeo(
                lon=[flon], lat=[flat],
                text=[f"<b>🏭 {fname}</b>"],
                hoverinfo="text",
                marker=dict(symbol="star", size=18, color="#f39c12",
                            line_width=1.5, line_color="white"),
                name=fname,
            ))
    fig.update_layout(
        geo=dict(scope="usa", projection_type="albers usa",
                 showland=True, landcolor="#f8f9fa",
                 showlakes=True, lakecolor="#cce5f5",
                 subunitcolor="#cccccc"),
        title=dict(text=f"US Shipping Heatmap — {map_metric}", font_color=BRAND),
        height=550, legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regional Bottleneck Summary")
    region_tbl = df.groupby("Region").agg(
        Avg_Lead   = ("Lead Time",        "mean"),
        Delay_Rate = ("Is Delayed",       "mean"),
        Orders     = ("Row ID",           "count"),
        Efficiency = ("Efficiency Score", "mean"),
    ).reset_index().sort_values("Avg_Lead", ascending=False)
    region_tbl["Delay_Rate"] = (region_tbl["Delay_Rate"] * 100).round(1).astype(str) + "%"
    region_tbl["Avg_Lead"]   = region_tbl["Avg_Lead"].round(2)
    region_tbl["Efficiency"] = region_tbl["Efficiency"].round(1)
    st.dataframe(
        region_tbl.style.background_gradient(subset=["Avg_Lead"], cmap="RdYlGn_r"),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHIP MODE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🚚 Ship Mode Performance Comparison")

    sm_grp = df.groupby("Ship Mode").agg(
        Avg_Lead    = ("Lead Time",        "mean"),
        Median_Lead = ("Lead Time",        "median"),
        Std_Lead    = ("Lead Time",        "std"),
        Delay_Rate  = ("Is Delayed",       "mean"),
        Orders      = ("Row ID",           "count"),
        Avg_Sales   = ("Sales",            "mean"),
        Efficiency  = ("Efficiency Score", "mean"),
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(sm_grp, x="Ship Mode", y="Avg_Lead",
                     color="Ship Mode", color_discrete_map=PALETTE,
                     error_y="Std_Lead",
                     title="Avg Lead Time by Ship Mode (with Std Dev)",
                     labels={"Avg_Lead": "Avg Lead Time (days)"})
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          title_font_color=BRAND)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(sm_grp, x="Ship Mode", y="Delay_Rate",
                     color="Ship Mode", color_discrete_map=PALETTE,
                     title="Delay Rate by Ship Mode",
                     labels={"Delay_Rate": "Delay Rate"})
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          title_font_color=BRAND)
        fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.violin(df, x="Ship Mode", y="Lead Time", color="Ship Mode",
                        color_discrete_map=PALETTE, box=True, points="outliers",
                        title="Lead Time Distribution (Violin)")
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          title_font_color=BRAND)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(sm_grp, x="Avg_Lead", y="Efficiency",
                         size="Orders", color="Ship Mode",
                         color_discrete_map=PALETTE, hover_name="Ship Mode",
                         size_max=50,
                         title="Lead Time vs Efficiency (bubble = order volume)",
                         labels={"Avg_Lead": "Avg Lead Time",
                                 "Efficiency": "Efficiency Score"})
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", title_font_color=BRAND)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ship Mode Summary Table")
    st.dataframe(
        sm_grp.round(3).style.background_gradient(subset=["Avg_Lead"], cmap="RdYlGn_r"),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ROUTE DRILL-DOWN
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🔬 Route Drill-Down")

    selected_route = st.selectbox(
        "Select a Route", sorted(df["Route"].unique()), key="drilldown_route"
    )
    route_df = df[df["Route"] == selected_route]
    st.markdown(f"**Route:** `{selected_route}` | **{len(route_df):,} orders**")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Avg Lead Time",    f"{route_df['Lead Time'].mean():.1f}d")
    r2.metric("Delay Rate",       f"{route_df['Is Delayed'].mean()*100:.1f}%")
    r3.metric("Efficiency Score", f"{route_df['Efficiency Score'].mean():.1f}")
    r4.metric("Avg Sales/Order",  f"${route_df['Sales'].mean():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(route_df, x="Lead Time", color="Ship Mode",
                           color_discrete_map=PALETTE,
                           title=f"Lead Time Distribution — {selected_route}")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", title_font_color=BRAND)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        prod_grp = route_df.groupby("Product Name")["Lead Time"].mean().reset_index()
        fig = px.bar(prod_grp.sort_values("Lead Time", ascending=False),
                     x="Lead Time", y="Product Name", orientation="h",
                     title="Avg Lead Time by Product",
                     color="Lead Time", color_continuous_scale="RdYlGn_r")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", title_font_color=BRAND,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    timeline = route_df.sort_values("Order Date").copy()
    timeline["Order #"] = range(1, len(timeline) + 1)
    fig = px.scatter(timeline, x="Order Date", y="Lead Time",
                     color="Ship Mode", color_discrete_map=PALETTE,
                     symbol="Is Delayed",
                     title="Order-Level Shipment Timeline",
                     hover_data=["Product Name", "Sales", "Gross Profit"])
    fig.add_hline(y=lt_thresh, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold ({lt_thresh}d)")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", title_font_color=BRAND)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View raw orders for this route"):
        st.dataframe(
            route_df[["Order ID", "Order Date", "Ship Date", "Ship Mode",
                       "Product Name", "Lead Time", "Is Delayed",
                       "Sales", "Gross Profit"]]
            .sort_values("Order Date").reset_index(drop=True),
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🤖 Machine Learning Insights")

    if not models:
        st.warning("⚠️ No trained models found — they will be generated on next restart.")
    else:
        ml_tab1, ml_tab2, ml_tab3 = st.tabs(["Regression", "Classification", "Clustering"])

        with ml_tab1:
            st.markdown("### Lead Time Regression")
            if "reg" in models:
                m   = models["reg"]
                res = m["results"]
                st.markdown(f"**Best Model:** `{m['model_name']}`")
                rows = [{"Model": k, "MAE": v["MAE"], "RMSE": v["RMSE"],
                         "R²": v["R2"], "CV R²": v["CV_R2"]}
                        for k, v in res.items()]
                st.dataframe(
                    pd.DataFrame(rows).set_index("Model").round(4),
                    use_container_width=True
                )
                for img in ["reg_feature_importance.png", "reg_actual_vs_predicted.png"]:
                    p = os.path.join(EDA_OUT, img)
                    if os.path.exists(p):
                        st.image(p)

        with ml_tab2:
            st.markdown("### Delay Classification")
            if "clf" in models:
                for img in ["clf_confusion_matrix.png", "clf_feature_importance.png"]:
                    p = os.path.join(EDA_OUT, img)
                    if os.path.exists(p):
                        st.image(p)

        with ml_tab3:
            st.markdown("### Route Clustering")
            if "clu" in models:
                clu          = models["clu"]
                route_df_clu = clu["route_df"]
                st.markdown(f"**Optimal k = {clu['best_k']}**")
                for c, lbl in clu["label_map"].items():
                    st.markdown(f"- Cluster {c}: {lbl}")
                fig = px.scatter(route_df_clu, x="PC1", y="PC2",
                                 color="Cluster Label", hover_name="Route",
                                 size="Total_Orders", size_max=30,
                                 title="Route Clusters (PCA 2D Projection)")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", title_font_color=BRAND)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    route_df_clu[["Route", "Cluster Label", "Avg_Lead_Time",
                                  "Delay_Rate", "Total_Orders", "Avg_Efficiency"]]
                    .sort_values("Avg_Lead_Time").round(3),
                    use_container_width=True
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🔮 Lead Time & Delay Predictor")
    st.markdown(
        "Use the trained ML models to predict shipping lead time "
        "and delay risk for a new order."
    )

    if not models:
        st.warning("⚠️ Models are being trained — please refresh the page in a moment.")
    else:
        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                p_ship_mode = st.selectbox("Ship Mode", df["Ship Mode"].unique())
                p_factory   = st.selectbox("Factory",   df["Factory"].unique())
                p_region    = st.selectbox("Region",    df["Region"].unique())
            with col2:
                p_division  = st.selectbox("Division",  df["Division"].unique())
                p_sales     = st.number_input("Sales ($)",  min_value=1.0,  value=10.0)
                p_units     = st.number_input("Units",      min_value=1,    value=2)
            with col3:
                p_profit    = st.number_input("Gross Profit ($)", min_value=0.0, value=5.0)
                p_cost      = st.number_input("Cost ($)",         min_value=0.0, value=3.0)
                p_month     = st.slider("Order Month", 1, 12, 6)

            submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

        if submitted:
            from sklearn.preprocessing import LabelEncoder

            NUMERIC_F = ["Sales", "Units", "Gross Profit", "Cost",
                         "Order Month", "Order Quarter", "Order DayOfWeek"]
            CATEG_F   = ["Ship Mode", "Factory", "Region", "Division"]

            input_raw = {
                "Ship Mode":       p_ship_mode,
                "Factory":         p_factory,
                "Region":          p_region,
                "Division":        p_division,
                "Sales":           p_sales,
                "Units":           p_units,
                "Gross Profit":    p_profit,
                "Cost":            p_cost,
                "Order Month":     p_month,
                "Order Quarter":   (p_month - 1) // 3 + 1,
                "Order DayOfWeek": 1,
            }

            def build_input(model_bundle):
                le_dict = model_bundle["le_dict"]
                X = [float(input_raw[f]) for f in NUMERIC_F]
                for f in CATEG_F:
                    le  = le_dict[f]
                    val = input_raw[f]
                    try:
                        enc = le.transform([val])[0]
                    except ValueError:
                        enc = 0
                    X.append(float(enc))
                return np.array(X).reshape(1, -1)

            if "reg" in models:
                X_arr   = build_input(models["reg"])
                pred_lt = models["reg"]["model"].predict(X_arr)[0]
                st.success(f"⏱ **Predicted Lead Time:** `{pred_lt:.1f} days`")

            if "clf" in models:
                X_arr   = build_input(models["clf"])
                pred_cl = models["clf"]["model"].predict(X_arr)[0]
                pred_pr = models["clf"]["model"].predict_proba(X_arr)[0]
                label   = "🔴 **DELAYED**" if pred_cl == 1 else "🟢 **EFFICIENT**"
                conf    = max(pred_pr) * 100
                st.info(f"🚦 **Delay Prediction:** {label}  (confidence: {conf:.1f}%)")
