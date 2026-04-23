# 🍬 Nassau Candy Distributor — Shipping Route Efficiency Dashboard

> A data-driven Streamlit dashboard for analysing factory-to-customer shipping performance across the United States, powered by Machine Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Dashboard Tabs](#-dashboard-tabs)
- [Machine Learning Models](#-machine-learning-models)
- [Dataset](#-dataset)
- [Factories & Products](#-factories--products)
- [KPI Reference](#-kpi-reference)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)

---

## 🔍 Overview

Nassau Candy Distributor ships products from **5 candy factories** to customers across **50 US states**. This project transforms raw order and shipment data into actionable route-level intelligence to:

- Identify the **fastest and slowest** shipping routes
- Detect **geographic bottlenecks** and delay-prone regions
- Compare performance across **4 ship modes**
- Predict **lead times and delays** using trained ML models

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 Route Overview | Top 10 best and worst performing routes with interactive charts |
| 🗺️ US Geo Heatmap | Interactive bubble map of delivery performance by state |
| 🚚 Ship Mode Analysis | Side-by-side comparison of Same Day, First Class, Second Class, Standard |
| 🔬 Route Drill-Down | Per-route timeline, product breakdown, and raw order table |
| 🤖 ML Insights | Regression, Classification, and Clustering results with visualisations |
| 🔮 Live Predictor | Predict lead time and delay risk for any new order |
| 🎛️ Interactive Filters | Filter by date, region, factory, ship mode, and lead-time threshold |

---

## 📁 Project Structure

```
nassau-candy/
│
├── data/
│   ├── Nassau_Candy_Distributor.csv    ← Raw input data (place here)
│   └── nassau_clean.csv                ← Generated after preprocessing
│
├── models/
│   ├── regression_model.pkl            ← Lead time regression model
│   ├── classification_model.pkl        ← Delay classification model
│   └── clustering_model.pkl            ← Route clustering model
│
├── outputs/
│   ├── eda/                            ← EDA charts (PNG)
│   └── ml/                             ← ML result charts (PNG)
│
├── scripts/
│   ├── 01_preprocessing.py             ← Data cleaning & feature engineering
│   ├── 02_eda.py                       ← Exploratory data analysis charts
│   ├── 03_ml_models.py                 ← Train regression, classification & clustering
│   └── 04_dashboard.py                 ← Streamlit dashboard (main app)
│
├── requirements.txt                    ← Python dependencies
└── README.md                           ← This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (comes with Python)
- joblib

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/nassau-candy-dashboard.git
cd nassau-candy-dashboard
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

> ⚠️ **Run these steps in order.** Each script depends on the previous one.

### Step 1 — Prepare your data

Place the raw CSV file in the `data/` folder:
```
data/Nassau_Candy_Distributor.csv
```

### Step 2 — Clean & engineer features

```bash
python scripts/reprocessing.py
```
✅ Creates `data/nassau_clean.csv`

### Step 3 — (Optional) Generate EDA charts

```bash
python scripts/eda.py
```
✅ Saves charts to `outputs/eda/`

### Step 4 — Train ML models

```bash
python scripts/ml_models.py
```
✅ Saves trained models to `models/` and charts to `outputs/ml/`
⏱ Takes 1–3 minutes depending on your machine.

### Step 5 — Launch the dashboard

```bash
streamlit run scripts/dashboard.py
```
✅ Opens automatically at **http://localhost:8501**

---

## 📊 Dashboard Tabs

| Tab | What it shows |
|---|---|
| **Overview** | Route performance leaderboard, lead time distributions, monthly trends |
| **Geo Map** | US bubble map — filter by Avg Lead Time, Delay Rate, Orders, or Efficiency |
| **Ship Mode** | Bar, violin and bubble charts comparing all 4 shipping methods |
| **Route Drill-Down** | Deep dive into any one route — timeline, product breakdown, raw data |
| **ML Insights** | Regression metrics, confusion matrix, cluster scatter plot |
| **Predictor** | Fill a form to get an instant lead time + delay prediction |

---

## 🤖 Machine Learning Models

### 1. Regression — Predict Lead Time
- **Models compared:** Ridge, Random Forest, Gradient Boosting
- **Target:** `Lead Time` (days)
- **Best model selected by:** CV R²
- **Output:** `models/regression_model.pkl`

### 2. Classification — Predict Delay
- **Models compared:** Random Forest, Gradient Boosting
- **Target:** `Is Delayed` (0 = on time, 1 = delayed)
- **Best model selected by:** CV F1 Score
- **Output:** `models/classification_model.pkl`

### 3. Clustering — Route Grouping
- **Algorithm:** K-Means with silhouette optimisation
- **Features:** Avg lead time, delay rate, order volume, efficiency score, profit
- **Output:** `models/clustering_model.pkl`
- **Cluster labels:** ⚡ Fast & Efficient | 🔴 High Delay Risk | 📦 High Volume

### Feature Set

| Type | Features |
|---|---|
| Numeric | Sales, Units, Gross Profit, Cost, Order Month, Quarter, Day of Week |
| Categorical | Ship Mode, Factory, Region, Division |

---

## 📦 Dataset

The dataset contains order and shipment records from Nassau Candy Distributor.

| Field | Description |
|---|---|
| Order Date | Date the order was placed |
| Ship Date | Date the order was shipped |
| Ship Mode | Shipping method (Same Day / First Class / Second Class / Standard) |
| State/Province | US state of the customer |
| Region | US region (West / East / Central / South) |
| Product Name | Name of the candy product |
| Sales | Total sales value ($) |
| Units | Number of units ordered |
| Gross Profit | Sales minus manufacturing cost |
| Lead Time | Calculated: Ship Date − Order Date (days) |
| Is Delayed | 1 if lead time exceeds ship-mode threshold, else 0 |
| Efficiency Score | Normalised 0–100 score (higher = faster relative to all routes) |

---

## 🏭 Factories & Products

| Factory | State | Products |
|---|---|---|
| Lot's O' Nuts | Arizona | Wonka Bar - Nutty Crunch Surprise, Fudge Mallows, Scrumdiddlyumptious |
| Wicked Choccy's | Georgia | Wonka Bar - Milk Chocolate, Triple Dazzle Caramel |
| Sugar Shack | Minnesota | Laffy Taffy, SweeTARTS, Nerds, Fun Dip, Fizzy Lifting Drinks |
| Secret Factory | Illinois | Everlasting Gobstopper, Lickable Wallpaper, Wonka Gum |
| The Other Factory | Tennessee | Hair Toffee, Kazookles |

---

## 📖 KPI Reference

| KPI | Formula / Meaning |
|---|---|
| Lead Time | Ship Date − Order Date (days) |
| Avg Lead Time | Mean lead time per route |
| Delay Rate | % of orders exceeding ship-mode threshold |
| Efficiency Score | `100 × (max_lt − lt) / (max_lt − min_lt)` |
| Profit Margin % | `Gross Profit / Sales × 100` |

### Delay Thresholds by Ship Mode

| Ship Mode | Threshold (days) |
|---|---|
| Same Day | 175 |
| First Class | 177 |
| Second Class | 179 |
| Standard Class | 181 |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: your feature description"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

*Built for Nassau Candy Distributor — Route Efficiency Analytics Project*