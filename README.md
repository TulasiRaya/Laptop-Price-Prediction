# 💻 LaptopPriceIQ — Intelligent Laptop Price Prediction Engine

> A production-grade Machine Learning pipeline that predicts laptop prices from hardware and software specifications, enabling data-driven competitive pricing for retail and e-commerce businesses.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)

---

## 📌 Problem Statement

Laptop pricing in the consumer electronics market is notoriously opaque. Retailers and procurement teams must manually track hundreds of feature combinations — CPU tier, display quality, storage configuration, GPU class, and brand premium — to price competitively. This process is slow, inconsistent, and error-prone.

**SmartTech Co.** needed a scalable, automated solution that could:
- Predict the fair market price of any laptop from its technical specifications
- Identify which hardware features are the strongest price drivers
- Generate instant price estimates for newly launched models not yet in any catalogue

Incorrect pricing directly translates to lost revenue (under-pricing) or lost customers (over-pricing). A reliable ML model removes this guesswork.

---

## 💡 Solution Overview

LaptopPriceIQ is a supervised regression pipeline built on 1,303 real-world laptop listings. It transforms raw, messy specification data (text strings, mixed units, compound fields) into structured numeric features, trains and evaluates five regression models head-to-head, then deploys the best performer — **Gradient Boosting Regressor** (R² = 0.821) — for real-time batch price inference on unseen laptops.

The pipeline is fully automated: raw CSV in → cleaned features → trained model → predicted prices out.

---

## ✨ Key Features

- **Advanced Feature Engineering** — Decomposes raw text columns (`ScreenResolution`, `Cpu`, `Memory`, `Gpu`, `OpSys`) into 17 structured numeric and categorical features, including derived metrics like PPI (Pixels Per Inch)
- **Multi-Model Benchmarking** — Trains and evaluates Linear Regression, SVR, Decision Tree, Random Forest, and Gradient Boosting in a unified sklearn Pipeline, with MAE / RMSE / R² comparison
- **Automated Preprocessing Pipeline** — `ColumnTransformer` applies `StandardScaler` to numeric features and `OneHotEncoder` (with unknown-category handling) to categorical features in a single, leakage-free step
- **Hyperparameter Optimization** — 5-fold `GridSearchCV` over `n_estimators`, `learning_rate`, and `max_depth` tunes the best model to peak performance
- **Feature Importance Interpretability** — Visualizes the top 20 features driving price, satisfying business explainability requirements
- **Batch Inference on New Data** — Accepts a separate `new_laptops.csv` and outputs `Predicted_Price` per record, including distribution visualization
- **Segment-Level Evaluation** — Measures model accuracy separately on budget (<₹50,000), mid-range, and premium (>₹1,00,000) laptop segments, and on lesser-known brands

---

## 🧠 Machine Learning Details

### Dataset

| Property | Value |
|---|---|
| Source file | `laptop.csv` |
| Raw records | 1,303 laptop listings |
| After cleaning | ~1,260 rows (removed 30 missing-value rows + invalid entries) |
| After outlier removal (IQR) | ~1,100–1,200 rows |
| Target variable | `Price` (continuous, INR) |
| Price range | ₹9,270 – ₹3,24,954 (mean ≈ ₹59,955) |
| Original columns | 13 (Company, TypeName, Inches, ScreenResolution, Cpu, Ram, Memory, Gpu, OpSys, Weight, Price) |
| Engineered features | 17 final features fed to models |

### Feature Engineering

| Raw Column | Engineered Output(s) |
|---|---|
| `ScreenResolution` | `Touchscreen` (binary), `IPS Technology` (binary), `X_res`, `Y_res`, `PPI` |
| `Cpu` | `Cpu_brand` (Intel i3/i5/i7/i9, AMD Ryzen 3/5/7, Apple Silicon, etc.) |
| `Memory` | `SSD` (GB), `HDD` (GB), `Flash_Storage` (GB), `Hybrid` (GB) |
| `Gpu` | `Gpu_brand` (Nvidia, AMD, Intel, Other) |
| `OpSys` | `OS_Category` (Windows, Mac, Linux, Chrome OS, No OS) |
| `Ram`, `Weight` | Stripped of units (`GB`, `kg`) → numeric |

### Models Evaluated

| Model | Notes |
|---|---|
| Linear Regression | Baseline; assumes linear feature relationships |
| Support Vector Regressor (SVR) | RBF kernel; effective on non-linear data |
| Decision Tree Regressor | High variance baseline; prone to overfitting |
| Random Forest Regressor | `n_estimators=200`; ensemble bagging approach |
| **Gradient Boosting Regressor** | **Best performer; sequential error-correcting ensemble** |

### Hyperparameter Tuning (GridSearchCV)

```
Parameter Grid:
  n_estimators:   [100, 200]
  learning_rate:  [0.05, 0.1, 0.2]
  max_depth:      [3, 5, 7]

Strategy: 5-fold cross-validation, scoring='r2', n_jobs=-1
```

### Evaluation Metrics (Best Model — Gradient Boosting)

| Metric | Value |
|---|---|
| **R² Score** | **0.821** |
| MAE | Reported per run |
| RMSE | Reported per run |
| Budget Segment R² | Computed separately |
| Premium Segment R² | Computed separately |
| Lesser-Known Brands R² | Computed separately |

> R² of 0.821 means the model explains ~82% of the variance in laptop prices from hardware specifications alone — a strong result for a domain with significant brand-premium noise.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA LAYER                               │
│  laptop.csv (1,303 rows × 13 cols)   new_laptops.csv (20 rows)      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA CLEANING                                  │
│  • Drop junk index columns (Unnamed: 0, Unnamed: 0.1)               │
│  • Drop 30 fully-null rows (dropna)                                 │
│  • Strip units: Ram "8GB"→8, Weight "2.1kg"→2.1                     │
│  • Remove invalid "?" entries in Inches / Weight                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                              │
│  ScreenResolution → Touchscreen, IPS, X_res, Y_res, PPI            │
│  Cpu text         → Cpu_brand (10 categories)                       │
│  Memory text      → SSD, HDD, Flash_Storage, Hybrid (GB)            │
│  Gpu text         → Gpu_brand (4 categories)                        │
│  OpSys text       → OS_Category (5 categories)                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               EXPLORATORY DATA ANALYSIS (EDA)                       │
│  • Price distribution (histplot + KDE)                              │
│  • Outlier detection via boxplot                                    │
│  • Price vs RAM scatter                                             │
│  • Brand counts, OS distribution                                    │
│  • Correlation heatmap (numeric features)                           │
│  • IQR-based outlier removal on Price                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                            │
│  ColumnTransformer:                                                 │
│    Numeric (12 cols)  → StandardScaler                              │
│    Categorical (5 cols) → OneHotEncoder(drop='first',               │
│                            handle_unknown='ignore')                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              TRAIN / TEST SPLIT  (80% / 20%, seed=42)               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               MODEL TRAINING & SELECTION                            │
│  5 models trained via sklearn.pipeline.Pipeline                     │
│  Ranked by R², MAE, RMSE → Gradient Boosting wins                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              HYPERPARAMETER TUNING (GridSearchCV)                   │
│  5-fold CV over 18 parameter combinations                           │
│  Best R² = 0.821                                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│           INFERENCE & INTERPRETABILITY                              │
│  • Feature importance plot (top 20)                                 │
│  • Batch prediction on new_laptops.csv                              │
│  • Segment analysis (budget / premium / lesser brands)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

**Language**
- Python 3.9+

**Core ML Framework**
- scikit-learn — Pipeline, ColumnTransformer, GridSearchCV, all regressors

**Data Manipulation**
- pandas — DataFrame operations, feature extraction, regex parsing
- NumPy — Numerical computations (PPI formula, RMSE)

**Visualization**
- Matplotlib — Base charting (bar charts, feature importance)
- Seaborn — Statistical plots (histplot, boxplot, countplot, heatmap, scatterplot)

**Development Environment**
- Google Colab (primary) / Jupyter Notebook (local compatible)

**Data**
- CSV format — `laptop.csv` (training), `new_laptops.csv` (inference)

---

## 📂 Project Structure

```
Capstone_project_ML/
│
├── ML_Project_LPP.ipynb          # Main notebook — full pipeline end-to-end
│
├── LPP_Complete_Code_Explanation.docx  # Line-by-line code documentation
│
├── ML_Project_LPP_pdf.pdf        # PDF export of the executed notebook
│
├── laptop.csv                    # Training dataset (1,303 laptop listings)
│                                 # Columns: Company, TypeName, Inches,
│                                 # ScreenResolution, Cpu, Ram, Memory,
│                                 # Gpu, OpSys, Weight, Price
│
└── new_laptops.csv               # Inference dataset (20 new laptops)
                                  # Pre-engineered features, no Price column
```

**Key notebook sections:**
- Cells 1–16: Library imports, data loading, structural inspection
- Cells 17–29: Data cleaning (drop columns, handle nulls, fix types)
- Cells 30–43: Feature engineering (5 raw columns → 12 derived features)
- Cells 44–55: EDA visualizations + IQR outlier removal
- Cells 56–68: Preprocessing setup + train/test split
- Cells 69–76: 5-model training loop + results comparison
- Cells 77–84: GridSearchCV tuning + feature importance plot
- Cells 85–103: New laptop inference + segment-level evaluation

---


**Expected output:**

```
Company     TypeName    Cpu_brand       Predicted_Price
Dell        Ultrabook   Intel Core i5   ₹48,230.00
```

### Checking Feature Importance

After tuning (Cell 84), the notebook renders a horizontal bar chart of the top 20 features. Key drivers typically include: RAM, SSD capacity, PPI, CPU brand tier, and GPU brand.

---

## 📊 Results / Output

### Model Comparison Summary

| Model | R² Score | Relative Performance |
|---|---|---|
| Linear Regression | Lowest | Cannot capture non-linear interactions |
| SVR | Moderate | Struggles without extensive feature scaling |
| Decision Tree | Moderate | Overfits; high variance |
| Random Forest | High | Strong ensemble; n_estimators=200 |
| **Gradient Boosting** | **0.821 (best)** | **Sequential correction of residual errors** |

### Post-Tuning Performance (Gradient Boosting + GridSearchCV)

- **R² Score: 0.821** — Model accounts for ~82% of price variance
- MAE and RMSE reported per training run (depend on the random state and filtered dataset size)

### Segment-Level Analysis

- **Budget segment** (< ₹50,000): Evaluated separately to check if model is calibrated at the low end
- **Premium segment** (> ₹1,00,000): Evaluated separately to verify performance on high-specification laptops
- **Lesser-known brands** (MSI, Fujitsu, Samsung, Xiaomi, Google): Dedicated evaluation to assess generalization beyond dominant brands

### Inference Output Example

Running `best_model.predict(new_laptops_df)` on the 20-record `new_laptops.csv` produces a `Predicted_Price` column, accompanied by a KDE distribution plot of all predicted prices.

---

## 🚧 Challenges & Learnings

**Data Quality Issues**
Raw `ScreenResolution`, `Cpu`, and `Memory` columns were unstructured text blobs requiring multi-step regex parsing. For example, `Memory` could contain values like `"128GB SSD + 1TB HDD"`, needing split-and-aggregate logic to extract four separate storage columns.

**Mixed Type Columns**
`Ram` and `Weight` were stored as strings with unit suffixes (`"8GB"`, `"2.1kg"`). A small number of rows contained `"?"` as the value — these had to be detected and removed before type conversion, otherwise the entire `astype()` call would silently corrupt data.

**One-Hot Encoding Unseen Categories**
At inference time, `new_laptops.csv` can contain brand or CPU values not seen during training. `handle_unknown='ignore'` in `OneHotEncoder` ensures these are silently zeroed out rather than throwing runtime errors — a critical production-readiness detail.

**Brand Premium vs. Specification Value**
Apple laptops command significantly higher prices than equivalent-spec Windows machines. The model captures this through brand one-hot encoding, but the "brand trust" premium (warranty, ecosystem, support) is an inherently non-technical factor that limits pure specification-based regression accuracy.

**Learnings**
- Feature engineering from raw text is often more impactful than model selection
- sklearn `Pipeline` prevents train/test leakage by applying preprocessing only after splitting
- GridSearchCV with `n_jobs=-1` is essential for practical hyperparameter search at this scale
- Segment-level evaluation reveals where a globally good R² may be hiding local weaknesses

---

## 🔮 Future Improvements

- **Web Scraping Integration** — Auto-refresh the training dataset from e-commerce platforms (Flipkart, Amazon) using `BeautifulSoup` or `Scrapy` to keep the model current as new hardware launches
- **Streamlit / Gradio UI** — Deploy a browser-based interface where users select laptop specs from dropdowns and receive an instant price estimate
- **Model Serialization** — Save the tuned pipeline with `joblib` or `pickle` for deployment without retraining
- **XGBoost / LightGBM** — Benchmark against more optimized gradient boosting implementations that may push R² above 0.85
- **SHAP Values** — Replace static feature importance with SHAP (SHapley Additive exPlanations) for per-prediction explainability, particularly useful for justifying pricing decisions to stakeholders
- **Cross-Region Pricing** — Extend the model to predict prices across different regional markets (US, EU, Asia) by incorporating currency-normalized price targets
- **Automated Retraining Pipeline** — Build a CI/CD-style workflow that retrains and re-evaluates the model whenever new data is available, with automatic promotion if R² improves

---

## 👤 Author

Built as a Capstone Machine Learning Project — March 2026

*Domain: Machine Learning · Regression · Feature Engineering · Pricing Analytics*
