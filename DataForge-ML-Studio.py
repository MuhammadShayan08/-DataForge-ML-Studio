# =====================================================================
#  DataForge ML Studio — All Features Free, No Login, No Admin
# =====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pycaret.classification import (
    setup as clf_setup, compare_models as clf_compare, 
    pull as clf_pull, save_model as clf_save,
)
from pycaret.regression import (
    setup as reg_setup, compare_models as reg_compare,
    pull as reg_pull, save_model as reg_save,
)
import warnings, time, io, os, gc
from datetime import datetime
warnings.filterwarnings("ignore")

st.set_page_config(page_title="DataForge ML Studio", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────
#  MEMORY-SAFE TRAINING CONFIG
# ─────────────────────────────────────────────
MAX_ROWS_TRAINING   = 5_000
MAX_ROWS_WARNING    = 2_000
SAMPLE_RANDOM_STATE = 42

ALL_CLF_MODELS = ["lr","dt","rf","et","ridge","knn","nb","ada","xgboost","lightgbm","catboost","gbc","lda"]
ALL_REG_MODELS = ["lr","dt","rf","et","ridge","lasso","knn","ada","en","xgboost","lightgbm","catboost","gbr","br"]

# ─────────────────────────────────────────────
#  NOTEBOOK GENERATOR
# ─────────────────────────────────────────────
import json as _json

def generate_notebook(df, target_col, problem_type, results_df, best_model_name,
                      top_score, metric_name, dataset_name, training_time, fold,
                      normalize, train_size,
                      eda_selections=None, pre_selections=None,
                      train_selections=None, export_selections=None):
    """
    User ke selections ke mutabiq notebook banata hai.
    eda_selections, pre_selections, train_selections, export_selections
    — dicts hain jisme True/False values hain.
    Agar None hain toh sab kuch include hoga (default behaviour).
    """
    # ── Default: agar koi selection nahi di toh sab ON ──
    eda  = eda_selections   or {"distributions":True,"correlation":True,"missing":True,"target_dist":True,"boxplots":True,"scatter_matrix":True,"cat_bars":True,"outlier_plot":True}
    pre  = pre_selections   or {"drop_dups":True,"handle_missing":True,"normalize":True,"remove_outliers":False}
    trn  = train_selections or {"model_table":True,"bar_chart":True,"radar_chart":True,"feature_importance":True}
    exp  = export_selections or {"save_model":True,"load_predict":True,"summary_table":True}

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    null_pct = round(df.isnull().sum().sum() / df.size * 100, 2)
    dup_cnt  = int(df.duplicated().sum())

    def code_cell(source, cell_id=""):
        return {"cell_type":"code","execution_count":None,
                "id":cell_id or source[:8].replace(" ","_"),
                "metadata":{},"outputs":[],"source":source.strip()}

    def md_cell(source, cell_id=""):
        return {"cell_type":"markdown",
                "id":cell_id or source[:8].replace(" ","_"),
                "metadata":{},"source":source.strip()}

    sp  = "clf" if problem_type == "classification" else "reg"
    pm  = "pycaret.classification" if problem_type == "classification" else "pycaret.regression"
    ncr = repr(num_cols)
    ccr = repr(cat_cols)

    try:
        rs = results_df.to_string(index=False)
    except Exception:
        rs = str(results_df)

    cells = []

    # ── Section 1: Title ──
    cells.append(md_cell(f"""# ⚡ DataForge ML Studio — Auto-Generated Notebook

> **Dataset:** `{dataset_name}` | **Target:** `{target_col}` | **Problem:** {problem_type.title()}
> **Best Model:** {best_model_name} | **Score ({metric_name}):** `{top_score:.4f}` | **CV Folds:** `{fold}`

Complete reproducible pipeline — run locally or on Google Colab.""", "title"))

    # ── Section 2: Imports ──
    cells.append(md_cell("## 📦 Section 1 — Install & Import", "s1"))
    cells.append(code_cell(f"""# !pip install pycaret plotly pandas numpy openpyxl
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from {pm} import (
    setup as {sp}_setup, compare_models as {sp}_compare,
    pull as {sp}_pull, save_model as {sp}_save, load_model as {sp}_load,
)
print("✅ Libraries imported!")""", "s1_imp"))

    # ── Section 3: Load data ──
    cells.append(md_cell("## 📂 Section 2 — Load Dataset", "s2"))
    cells.append(code_cell(f"""# df = pd.read_csv("your_file.csv")
# df = pd.read_excel("your_file.xlsx")
TARGET = "{target_col}"
print(f"Shape: {{df.shape}}, Target: {{TARGET}}")
df.head()""", "s2_load"))

    # ── Section 4: Dataset overview ──
    cells.append(md_cell("## 🔍 Section 3 — Dataset Overview", "s3"))
    cells.append(code_cell(f"""print(f"Rows: {len(df):,} | Cols: {len(df.columns)} | Missing: {null_pct}% | Dups: {dup_cnt}")
df.info()""", "s3_info"))
    cells.append(code_cell("df.describe(include='all').round(3)", "s3_desc"))

    # ── Section 5: EDA — user selected charts only ──
    eda_cells_added = False

    if eda.get("missing"):
        if not eda_cells_added:
            cells.append(md_cell("## 🧬 Section 4 — EDA (Exploratory Data Analysis)", "s4"))
            eda_cells_added = True
        cells.append(code_cell("""# Missing values chart
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    fig = px.bar(x=missing.index, y=missing.values, title="Missing Values per Column",
                 color=missing.values, color_continuous_scale=["#4ade80","#fbbf24","#f87171"])
    fig.update_layout(template="plotly_dark", height=350)
    fig.show()
else:
    print("✅ No missing values found!")""", "s4_miss"))

    if eda.get("distributions") and num_cols:
        if not eda_cells_added:
            cells.append(md_cell("## 🧬 Section 4 — EDA (Exploratory Data Analysis)", "s4"))
            eda_cells_added = True
        cells.append(code_cell(f"""# Numeric distributions
num_cols = {ncr}
cols_to_plot = num_cols[:9]
n_c = min(3, len(cols_to_plot))
n_r = (len(cols_to_plot) + n_c - 1) // n_c
fig = make_subplots(rows=n_r, cols=n_c, subplot_titles=cols_to_plot)
for i, col in enumerate(cols_to_plot):
    fig.add_trace(go.Histogram(x=df[col], name=col, marker_color="#4ade80", opacity=0.8),
                  row=i//n_c+1, col=i%n_c+1)
fig.update_layout(template="plotly_dark", height=280*n_r, title_text="Distributions", showlegend=False)
fig.show()""", "s4_hist"))

    if eda.get("boxplots") and num_cols:
        if not eda_cells_added:
            cells.append(md_cell("## 🧬 Section 4 — EDA", "s4"))
            eda_cells_added = True
        cells.append(code_cell(f"""# Box plots
num_cols = {ncr}
cols_bp = num_cols[:6]
n_c = min(3, len(cols_bp))
n_r = (len(cols_bp) + n_c - 1) // n_c
fig = make_subplots(rows=n_r, cols=n_c, subplot_titles=cols_bp)
for i, col in enumerate(cols_bp):
    fig.add_trace(go.Box(y=df[col], name=col, marker_color="#60a5fa"), row=i//n_c+1, col=i%n_c+1)
fig.update_layout(template="plotly_dark", height=280*n_r, title_text="Box Plots", showlegend=False)
fig.show()""", "s4_box"))

    if eda.get("correlation") and len(num_cols) >= 2:
        if not eda_cells_added:
            cells.append(md_cell("## 🧬 Section 4 — EDA", "s4"))
            eda_cells_added = True
        cells.append(code_cell(f"""# Correlation heatmap
corr = df[{ncr}[:15]].corr().round(2)
fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
    colorscale=[[0,"#f87171"],[0.5,"#1c1c1c"],[1,"#4ade80"]], zmid=0,
    text=corr.values.round(2), texttemplate="%{{text}}", textfont=dict(size=9)))
fig.update_layout(template="plotly_dark", height=520, title="Correlation Heatmap")
fig.show()""", "s4_corr"))

    if eda.get("scatter_matrix") and len(num_cols) >= 2:
        if not eda_cells_added:
            cells.append(md_cell("## 🧬 Section 4 — EDA", "s4"))
            eda_cells_added = True
        cells.append(code_cell(f"""# Scatter matrix
import plotly.figure_factory as ff
scatter_cols = {ncr}[:5]
fig = px.scatter_matrix(df[scatter_cols], title="Scatter Matrix",
                        color_discrete_sequence=["#4ade80"])
fig.update_layout(template="plotly_dark", height=600)
fig.show()""", "s4_scat"))

    if eda.get("target_dist"):
        if not eda_cells_added:
            cells.append(md_cell("## 🧬 Section 4 — EDA", "s4"))
            eda_cells_added = True
        cells.append(code_cell(f"""# Target column distribution
target_series = df["{target_col}"]
if "{problem_type}" == "classification":
    vc = target_series.value_counts()
    fig = px.pie(values=vc.values, names=vc.index.astype(str),
                 title="Target Distribution — {target_col}",
                 color_discrete_sequence=["#4ade80","#60a5fa","#c084fc","#fbbf24","#f87171"])
else:
    fig = px.histogram(df, x="{target_col}", nbins=40,
                       title="Target Distribution — {target_col}",
                       color_discrete_sequence=["#4ade80"])
fig.update_layout(template="plotly_dark", height=380)
fig.show()""", "s4_tgt"))

    if eda.get("cat_bars") and cat_cols:
        if not eda_cells_added:
            cells.append(md_cell("## 🧬 Section 4 — EDA", "s4"))
            eda_cells_added = True
        cells.append(code_cell(f"""# Categorical column bar charts
cat_cols = {ccr}
for col in cat_cols[:4]:
    vc = df[col].value_counts().head(12)
    fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Value Counts — {{col}}",
                 color=vc.values, color_continuous_scale=["#60a5fa","#4ade80"])
    fig.update_layout(template="plotly_dark", height=320, showlegend=False, coloraxis_showscale=False)
    fig.show()""", "s4_cat"))

    if eda.get("outlier_plot") and num_cols:
        if not eda_cells_added:
            cells.append(md_cell("## 🧬 Section 4 — EDA", "s4"))
            eda_cells_added = True
        cells.append(code_cell(f"""# Outlier detection (IQR method)
num_cols = {ncr}
outlier_info = []
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    outlier_info.append({{"column": col, "outliers": int(n_out), "pct": round(n_out/len(df)*100, 2)}})
out_df = pd.DataFrame(outlier_info).sort_values("outliers", ascending=False)
fig = px.bar(out_df, x="column", y="pct", title="Outlier % per Column (IQR method)",
             color="pct", color_continuous_scale=["#4ade80","#fbbf24","#f87171"])
fig.update_layout(template="plotly_dark", height=380)
fig.show()
print(out_df.to_string(index=False))""", "s4_out"))

    # ── Section 6: Preprocessing — user selected steps only ──
    pre_steps = []
    if pre.get("drop_dups"):
        pre_steps.append("""# Drop duplicate rows
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicates removed: {before - len(df)} | Shape: {df.shape}")""")

    if pre.get("handle_missing"):
        pre_steps.append("""# Handle missing values — show summary
nc = df.isnull().sum()
nc = nc[nc > 0]
print("Missing cols before:", nc.to_string() if len(nc) else "None ✅")
# PyCaret handles imputation automatically in setup()""")

    if pre.get("normalize"):
        pre_steps.append(f"""# Normalization — handled by PyCaret setup (normalize={normalize})
# Will be applied via {sp}_setup(normalize=True)
print("✅ Normalization: will be applied in PyCaret setup")""")

    if pre.get("remove_outliers") and problem_type == "regression":
        pre_steps.append("""# Remove outliers (regression only — IQR method)
from scipy import stats
num_cols_pre = df.select_dtypes(include='number').columns.drop('target', errors='ignore')
z_scores = stats.zscore(df[num_cols_pre].fillna(df[num_cols_pre].median()))
df = df[(abs(z_scores) < 3).all(axis=1)].reset_index(drop=True)
print(f"After outlier removal: {df.shape}")""")

    if pre_steps:
        cells.append(md_cell("## 🧹 Section 5 — Preprocessing", "s5"))
        for i, step_code in enumerate(pre_steps):
            cells.append(code_cell(step_code, f"s5_pre{i}"))

    # ── Section 7: Training ──
    cells.append(md_cell(f"""## ⚙️ Section 6 — AutoML Training (PyCaret)

Config: **{problem_type}** | train_size=`{train_size}` | fold=`{fold}` | normalize=`{normalize}`""", "s6"))

    cells.append(code_cell(f"""{sp}_setup(data=df, target="{target_col}", train_size={train_size},
           fold={fold}, normalize={normalize}, verbose=True, session_id=42, n_jobs=-1)
print("✅ PyCaret setup complete!")""", "s6_setup"))

    cells.append(code_cell(f"""# ☕ This might take a few minutes
best_model = {sp}_compare(verbose=True, n_select=1)
results = {sp}_pull()
print("\\n✅ Done! Best model:", best_model)""", "s6_cmp"))

    # ── Section 8: Results — user selected charts only ──
    results_added = False

    if trn.get("model_table"):
        if not results_added:
            cells.append(md_cell("## 🏆 Section 7 — Results & Visualizations", "s7"))
            results_added = True
        cells.append(code_cell(f"""# All models comparison table
print("DataForge session results:")
print(\"\"\"{rs}\"\"\")
results_df = {sp}_pull()
results_df""", "s7_res"))

    if trn.get("bar_chart"):
        if not results_added:
            cells.append(md_cell("## 🏆 Section 7 — Results & Visualizations", "s7"))
            results_added = True
        cells.append(code_cell(f"""# Top models bar chart
results_df = {sp}_pull()
mc = results_df.select_dtypes(include='number').columns[0]
ml = results_df.columns[0]
top6 = results_df.head(6)
colors = ["#4ade80"] + ["#1c1c1c"] * 5
fig = go.Figure(go.Bar(x=top6[mc], y=top6[ml], orientation="h",
    marker_color=colors, text=top6[mc].round(4), textposition="inside",
    textfont=dict(size=11, color="white")))
fig.update_layout(template="plotly_dark", height=380,
    title=f"Top Models — {{mc}}", yaxis=dict(autorange="reversed"))
fig.show()""", "s7_bar"))

    if trn.get("radar_chart"):
        if not results_added:
            cells.append(md_cell("## 🏆 Section 7 — Results & Visualizations", "s7"))
            results_added = True
        cells.append(code_cell(f"""# Best model metrics radar chart
results_df = {sp}_pull()
nm = results_df.select_dtypes(include='number').columns[:6].tolist()
bv = results_df.iloc[0][nm]
norm = (bv - bv.min()) / (bv.max() - bv.min() + 1e-9)
fig = go.Figure(go.Scatterpolar(
    r=list(norm.values)+[norm.values[0]], theta=list(norm.index)+[norm.index[0]],
    fill="toself", fillcolor="rgba(74,222,128,0.18)",
    line=dict(color="#4ade80", width=2.5), marker=dict(size=6, color="#4ade80")))
fig.update_layout(template="plotly_dark", height=400,
    title="Best Model Metrics — {best_model_name}",
    polar=dict(radialaxis=dict(visible=True, range=[0,1])))
fig.show()""", "s7_radar"))

    if trn.get("feature_importance"):
        if not results_added:
            cells.append(md_cell("## 🏆 Section 7 — Results & Visualizations", "s7"))
            results_added = True
        cells.append(code_cell(f"""# Feature importance plot
try:
    import matplotlib.pyplot as plt
    from pycaret.{problem_type} import plot_model
    plot_model(best_model, plot="feature", display_format="streamlit")
except Exception:
    # Manual feature importance via sklearn
    try:
        importances = best_model.feature_importances_
        feature_names = df.drop(columns=["{target_col}"]).columns.tolist()
        fi_df = pd.DataFrame({{"feature": feature_names, "importance": importances}})
        fi_df = fi_df.sort_values("importance", ascending=True).tail(15)
        fig = px.bar(fi_df, x="importance", y="feature", orientation="h",
                     title="Feature Importance — {best_model_name}",
                     color="importance", color_continuous_scale=["#60a5fa","#4ade80"])
        fig.update_layout(template="plotly_dark", height=450)
        fig.show()
    except Exception as e:
        print(f"Feature importance not available for this model: {{e}}")""", "s7_feat"))

    # ── Section 9: Export — user selected options only ──
    export_added = False

    if exp.get("save_model"):
        if not export_added:
            cells.append(md_cell("## 💾 Section 8 — Save & Load Model", "s8"))
            export_added = True
        cells.append(code_cell(f"""{sp}_save(best_model, "best_model_dataforge")
print("✅ Saved: best_model_dataforge.pkl")""", "s8_save"))

    if exp.get("load_predict"):
        if not export_added:
            cells.append(md_cell("## 💾 Section 8 — Save & Load Model", "s8"))
            export_added = True
        cells.append(code_cell(f"""loaded_model = {sp}_load("best_model_dataforge")
print("✅ Loaded:", type(loaded_model))

# Make predictions on new data:
# from {pm} import predict_model
# df_new = df.drop(columns=[TARGET]).head(5)
# preds = predict_model(loaded_model, data=df_new)
# print(preds)""", "s8_load"))

    if exp.get("summary_table"):
        cells.append(md_cell(f"""## 📋 Section 9 — Session Summary

| Property | Value |
|---|---|
| Dataset | `{dataset_name}` |
| Target | `{target_col}` |
| Problem | {problem_type.title()} |
| Best Model | **{best_model_name}** |
| Score ({metric_name}) | `{top_score:.4f}` |
| CV Folds | {fold} |
| Training Time | {training_time:.1f}s |
| Train Split | {int(train_size*100)}% / {int((1-train_size)*100)}% |
| Normalize | {normalize} |

**EDA included:** {", ".join(k for k,v in (eda or {}).items() if v) or "None"}
**Preprocessing:** {", ".join(k for k,v in (pre or {}).items() if v) or "None"}
**Result charts:** {", ".join(k for k,v in (trn or {}).items() if v) or "None"}

> Generated by **DataForge ML Studio** ⚡ — Notebook Builder""", "s9"))

    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
            "language_info": {"name":"python","version":"3.10.0"}
        },
        "cells": cells
    }
    return _json.dumps(notebook, indent=2, ensure_ascii=False).encode("utf-8")


# ─────────────────────────────────────────────
#  MEMORY HELPERS
# ─────────────────────────────────────────────
def get_memory_usage_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

def force_gc():
    gc.collect(); gc.collect()
    try:
        import ctypes
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def smart_sample(df, target_col, max_rows=MAX_ROWS_TRAINING):
    if len(df) <= max_rows:
        return df
    try:
        target_series = df[target_col]
        if target_series.dtype == "object" or target_series.nunique() <= 20:
            class_counts = target_series.value_counts()
            valid_classes = class_counts[class_counts >= 2].index
            df_filtered = df[target_series.isin(valid_classes)]
            if len(df_filtered) >= max_rows:
                from sklearn.model_selection import train_test_split
                _, sampled = train_test_split(
                    df_filtered, test_size=max_rows / len(df_filtered),
                    stratify=df_filtered[target_col], random_state=SAMPLE_RANDOM_STATE
                )
                return sampled.reset_index(drop=True)
    except Exception:
        pass
    return df.sample(n=max_rows, random_state=SAMPLE_RANDOM_STATE).reset_index(drop=True)


def safe_fold_count(df, target_col, requested_fold, problem_type):
    if problem_type != "classification":
        return requested_fold
    try:
        min_class_count = df[target_col].value_counts().min()
        safe_fold = min(requested_fold, int(min_class_count))
        safe_fold = max(2, safe_fold)
        return safe_fold
    except Exception:
        return max(2, requested_fold)


def drop_rare_classes(df, target_col, min_count=2):
    try:
        class_counts = df[target_col].value_counts()
        rare = class_counts[class_counts < min_count].index.tolist()
        if rare:
            df_clean = df[~df[target_col].isin(rare)].reset_index(drop=True)
            warn = (
                f"\u26a0\ufe0f **{len(rare)} rare class(es) removed** (less than {min_count} samples): "
                f"`{'`, `'.join([str(r) for r in rare[:5]])}`. "
                f"Cross-validation ke liye minimum {min_count} samples per class zaroori hain."
            )
            return df_clean, warn
    except Exception:
        pass
    return df, None


def run_memory_safe_training(df, target_col, problem_type, train_size, fold,
                              normalize, remove_out, max_models=None):
    warnings_list = []
    t0 = time.time()

    if problem_type == "classification":
        df, rare_warn = drop_rare_classes(df, target_col, min_count=2)
        if rare_warn:
            warnings_list.append(rare_warn)
        if len(df) < 10:
            raise ValueError(
                "Dataset mein bahut kam samples hain. "
                "Target column sahi select kiya hai?"
            )

    original_rows = len(df)
    if original_rows > MAX_ROWS_TRAINING:
        df_train = smart_sample(df, target_col, MAX_ROWS_TRAINING)
        warnings_list.append(
            f"\u26a0\ufe0f Dataset {original_rows:,} rows — auto-sampled to **{MAX_ROWS_TRAINING:,} rows**."
        )
    elif original_rows > MAX_ROWS_WARNING:
        df_train = df.copy()
        warnings_list.append(
            f"\U0001f4a1 Dataset {original_rows:,} rows — training chalegi lekin agar crash ho toh {MAX_ROWS_WARNING:,} rows tak chota karo."
        )
    else:
        df_train = df.copy()

    safe_fold = safe_fold_count(df_train, target_col, fold, problem_type)
    if safe_fold != fold:
        warnings_list.append(
            f"\u26a0\ufe0f **CV Folds {fold} se {safe_fold} reduce** — kuch classes mein kam samples hain. "
            f"Training {safe_fold}-fold ke saath continue ho rahi hai."
        )

    include_models = ALL_CLF_MODELS if problem_type == "classification" else ALL_REG_MODELS

    if max_models and max_models < len(include_models):
        include_models = include_models[:max_models]

    mem_before = get_memory_usage_mb()
    if mem_before > 400:
        force_gc()

    setup_kwargs = dict(
        data=df_train, target=target_col,
        train_size=float(train_size), fold=int(safe_fold),
        normalize=normalize, verbose=False, html=False,
        session_id=42, n_jobs=1, use_gpu=False,
    )
    if remove_out and problem_type == "regression" and len(df_train) > 100:
        setup_kwargs["remove_outliers"] = True

    try:
        if problem_type == "classification":
            clf_setup(**setup_kwargs)
            pull_fn, save_fn, cmp_fn = clf_pull, clf_save, clf_compare
        else:
            reg_setup(**setup_kwargs)
            pull_fn, save_fn, cmp_fn = reg_pull, reg_save, reg_compare
    except Exception as e:
        err = str(e).lower()
        if "memory" in err or "killed" in err:
            raise MemoryError(f"Setup mein memory khatam — dataset {MAX_ROWS_WARNING:,} rows se kam karo.")
        if "least populated" in err or "stratif" in err or "minimum number of groups" in err:
            raise ValueError(
                "Cross-validation error: Kuch target classes mein bohot kam samples hain.\n\n"
                "Solutions:\n"
                "- CV Folds slider ko 2 par set karo\n"
                "- Un rows ko dataset se hata do jinka target value sirf 1-2 baar aata hai\n"
                "- Ya ek alag target column choose karo"
            )
        raise

    force_gc()

    try:
        best = cmp_fn(verbose=False, n_select=1, include=include_models, errors="ignore")
        results = pull_fn()
    except MemoryError:
        force_gc()
        light = ["lr","dt","ridge"]
        warnings_list.append("⚠️ Memory issue — sirf 3 lightest models se try kar raha hoon.")
        best = cmp_fn(verbose=False, n_select=1, include=light)
        results = pull_fn()
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ["memory","killed","oom","cannot allocate"]):
            raise MemoryError("Model comparison mein memory khatam — dataset chota karo.")
        raise

    try:
        save_fn(best, "best_model")
    except Exception:
        pass

    force_gc()
    elapsed = time.time() - t0
    return best, results, elapsed, warnings_list, len(df_train)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for k in ["data","problem_type","best_model","results","training_time","dataset_name","cv_fold"]:
    if k not in st.session_state:
        st.session_state[k] = None
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

T = st.session_state.theme

# ─────────────────────────────────────────────
#  THEME PALETTES
# ─────────────────────────────────────────────
if T == "dark":
    BG = "#000000"; BG2 = "#0d0d0d"; BG3 = "#141414"; BG4 = "#1c1c1c"
    BORDER = "#222222"; TEXT1 = "#f9fafb"; TEXT2 = "#9ca3af"; TEXT3 = "#6b7280"
    ACCENT1 = "#4ade80"; ACCENT2 = "#60a5fa"; ACCENT3 = "#c084fc"
    ACCENTR = "#f87171"; ACCENTY = "#fbbf24"
    HDR_BG = "linear-gradient(135deg,#000000 0%,#0d0d0d 60%,#000000 100%)"
    HDR_BORDER = "rgba(74,222,128,0.25)"
    BTN_BG = "linear-gradient(135deg,#16a34a,#22c55e)"; BTN_GLOW = "rgba(74,222,128,0.40)"
    TAB_SEL = "linear-gradient(135deg,#16a34a,#22c55e)"
    CARD_BG = "#0d0d0d"; CHART_TEMPLATE = "plotly_dark"
    CHART_PAPER = "rgba(0,0,0,0)"; CHART_FONT = "#9ca3af"; CHART_GRID = "#1c1c1c"
    GLOW_DIV = "linear-gradient(90deg,transparent,#4ade80,#60a5fa,transparent)"
    HERO_H1_GRAD = "linear-gradient(135deg,#4ade80 0%,#60a5fa 50%,#c084fc 100%)"
else:
    BG = "#f8f4ff"; BG2 = "#ffffff"; BG3 = "#ede9fe"; BG4 = "#ddd6fe"
    BORDER = "#c4b5fd"; TEXT1 = "#1e0a3c"; TEXT2 = "#5b21b6"; TEXT3 = "#7c3aed"
    ACCENT1 = "#7c3aed"; ACCENT2 = "#2563eb"; ACCENT3 = "#0891b2"
    ACCENTR = "#dc2626"; ACCENTY = "#d97706"
    HDR_BG = "linear-gradient(135deg,#1e0a3c 0%,#4c1d95 50%,#2e1065 100%)"
    HDR_BORDER = "rgba(167,139,250,0.4)"
    BTN_BG = "linear-gradient(135deg,#5b21b6,#7c3aed)"; BTN_GLOW = "rgba(124,58,237,0.40)"
    TAB_SEL = "linear-gradient(135deg,#5b21b6,#7c3aed)"
    CARD_BG = "#ffffff"; CHART_TEMPLATE = "plotly_white"
    CHART_PAPER = "rgba(0,0,0,0)"; CHART_FONT = "#5b21b6"; CHART_GRID = "#ddd6fe"
    GLOW_DIV = "linear-gradient(90deg,transparent,#7c3aed,#2563eb,transparent)"
    HERO_H1_GRAD = "linear-gradient(135deg,#7c3aed 0%,#2563eb 50%,#0891b2 100%)"

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
*,*::before,*::after{{transition:background 0.65s ease,background-color 0.65s ease,color 0.45s ease,border-color 0.45s ease,box-shadow 0.45s ease !important;font-family:'Inter',-apple-system,sans-serif;}}
#MainMenu{{visibility:hidden;}}footer{{visibility:hidden;}}
.block-container{{padding-top:1.5rem !important;max-width:1400px;}}
html,body{{background:{BG} !important;color:{TEXT1} !important;}}
.main,.block-container,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"]{{background:{BG} !important;}}
section[data-testid="stSidebar"]{{background:{"linear-gradient(180deg,#0a0a0a 0%,#111111 100%)" if T=="dark" else "#f7f7fb"} !important;border-right:{"1px solid #222222" if T=="dark" else "2px solid #d0d0d0"} !important;box-shadow:{"4px 0 20px rgba(0,0,0,0.6)" if T=="dark" else "4px 0 16px rgba(0,0,0,0.10)"} !important;min-height:100vh !important;}}
section[data-testid="stSidebar"]>div{{background:{"transparent" if T=="dark" else "#f7f7fb"} !important;}}
section[data-testid="stSidebar"] *{{color:{"#d1fae5" if T=="dark" else "#111111"} !important;}}
section[data-testid="stSidebar"] .stButton>button{{background:{"linear-gradient(135deg,#16a34a,#22c55e)" if T=="dark" else "linear-gradient(135deg,#5b21b6,#7c3aed)"} !important;color:#ffffff !important;box-shadow:{"0 4px 14px rgba(74,222,128,0.35)" if T=="dark" else "0 4px 14px rgba(124,58,237,0.35)"} !important;}}
[data-testid="stFileUploader"]{{background:{"#0f0f0f" if T=="dark" else "#f3eeff"} !important;border-radius:14px !important;padding:4px !important;}}
[data-testid="stFileUploader"]>div{{background:{"#0d0d0d" if T=="dark" else "#ede9fe"} !important;border:2px dashed {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(124,58,237,0.40)"} !important;border-radius:12px !important;color:{"#4ade80" if T=="dark" else "#7c3aed"} !important;}}
[data-testid="stFileUploader"] *{{color:{"#4ade80" if T=="dark" else "#7c3aed"} !important;background:transparent !important;}}
[data-testid="stFileUploaderDropzoneInstructions"] *{{color:{"rgba(74,222,128,0.55)" if T=="dark" else "rgba(124,58,237,0.55)"} !important;}}
[data-testid="stFileUploader"] button{{background:{"#1a1a1a" if T=="dark" else "#ede9fe"} !important;border:1px solid {"rgba(74,222,128,0.30)" if T=="dark" else "rgba(124,58,237,0.30)"} !important;color:{"#9ca3af" if T=="dark" else "#7c3aed"} !important;}}
.vibe-header{{position:relative;padding:2.5rem 3rem;border-radius:20px;margin-bottom:2rem;overflow:hidden;background:{HDR_BG};border:1px solid {HDR_BORDER};box-shadow:{"0 0 40px rgba(74,222,128,0.12)" if T=="dark" else "0 8px 40px rgba(124,58,237,0.30)"}}}
.vibe-header::before{{content:'';position:absolute;inset:0;background:{"radial-gradient(ellipse 80% 60% at 10% 50%,rgba(74,222,128,0.15) 0%,transparent 60%)" if T=="dark" else "radial-gradient(ellipse 80% 60% at 10% 50%,rgba(167,139,250,0.25) 0%,transparent 60%)"};animation:pulseGlow 6s ease-in-out infinite alternate;}}
@keyframes pulseGlow{{from{{opacity:.6}}to{{opacity:1}}}}
.vibe-header h1{{font-size:2.8rem;font-weight:900;margin:0;letter-spacing:-.03em;line-height:1.1;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}}
.vibe-header .tagline{{font-size:1rem;margin-top:.6rem;font-weight:400;color:{"#a7f3d0" if T=="dark" else "#ddd6fe"};}}
.vibe-header .pill-row{{display:flex;gap:.5rem;margin-top:1rem;flex-wrap:wrap;}}
.pill{{display:inline-flex;align-items:center;gap:.3rem;padding:.25rem .8rem;border-radius:999px;font-size:.7rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;}}
.pill-green{{background:{"rgba(74,222,128,0.15)" if T=="dark" else "rgba(167,139,250,0.20)"};color:{"#4ade80" if T=="dark" else "#ddd6fe"};border:1px solid {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(167,139,250,0.40)"};}}
.pill-blue{{background:{"rgba(96,165,250,0.12)" if T=="dark" else "rgba(196,181,253,0.20)"};color:{"#60a5fa" if T=="dark" else "#c4b5fd"};border:1px solid {"rgba(96,165,250,0.30)" if T=="dark" else "rgba(196,181,253,0.40)"};}}
.pill-purple{{background:{"rgba(192,132,252,0.12)" if T=="dark" else "rgba(216,180,254,0.20)"};color:{"#c084fc" if T=="dark" else "#e9d5ff"};border:1px solid {"rgba(192,132,252,0.30)" if T=="dark" else "rgba(216,180,254,0.40)"};}}
.stat-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-bottom:1.5rem;}}
.stat-card{{background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;padding:1.25rem 1.5rem;position:relative;overflow:hidden;cursor:default;}}
.stat-card:hover{{border-color:{ACCENT1};transform:translateY(-4px);box-shadow:{"0 0 28px rgba(74,222,128,0.22)" if T=="dark" else "0 8px 28px rgba(124,58,237,0.22)"}}}
.stat-card .bar{{position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{ACCENT1},{ACCENT2});transform:scaleX(0);transform-origin:left;border-radius:2px;}}
.stat-card:hover .bar{{transform:scaleX(1);}}
.stat-card .label{{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:{TEXT3};margin-bottom:.4rem;}}
.stat-card .value{{font-size:2rem;font-weight:800;color:{TEXT1};font-variant-numeric:tabular-nums;line-height:1;}}
.stat-card .sub{{font-size:.72rem;color:{TEXT3};margin-top:.3rem;}}
.stat-card.good .value{{color:{ACCENT1};}}
.stat-card.warn .value{{color:{ACCENTY};}}
.stat-card.danger .value{{color:{ACCENTR};}}
.section-head{{display:flex;align-items:center;gap:.75rem;margin:2rem 0 1rem;}}
.section-head .icon-wrap{{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;background:{"linear-gradient(135deg,rgba(74,222,128,0.15),rgba(96,165,250,0.12))" if T=="dark" else "linear-gradient(135deg,rgba(124,58,237,0.18),rgba(37,99,235,0.12))"};border:1px solid {"rgba(74,222,128,0.3)" if T=="dark" else "rgba(124,58,237,0.30)"};}}
.section-head h3{{margin:0;font-size:1.1rem;font-weight:700;color:{TEXT1};}}
.feature-card{{background:{CARD_BG};border:1px solid {BORDER};border-radius:18px;padding:1.75rem;position:relative;overflow:hidden;}}
.feature-card:hover{{border-color:{ACCENT1};transform:translateY(-3px);box-shadow:{"0 0 28px rgba(74,222,128,0.18)" if T=="dark" else "0 10px 32px rgba(124,58,237,0.18)"}}}
.feature-card .fc-icon{{font-size:2.2rem;margin-bottom:.75rem;}}
.feature-card h3{{margin:0 0 .5rem;font-size:1rem;font-weight:700;color:{TEXT1};}}
.feature-card p{{margin:0;font-size:.875rem;color:{TEXT2};line-height:1.6;}}
.sidebar-section{{background:{"rgba(255,255,255,0.05)" if T=="dark" else "#ffffff"};border:1px solid {"rgba(255,255,255,0.08)" if T=="dark" else "#dddddd"};border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem;}}
.sidebar-title{{font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{"#4ade80" if T=="dark" else "#7c3aed"};margin-bottom:.75rem;}}
.target-card{{border-radius:16px;padding:1.5rem;margin:1rem 0;border:1px solid;display:grid;grid-template-columns:auto 1fr;gap:1rem;align-items:center;}}
.target-card.clf{{background:{"rgba(74,222,128,0.06)" if T=="dark" else "rgba(124,58,237,0.06)"};border-color:{"rgba(74,222,128,0.35)" if T=="dark" else "rgba(124,58,237,0.35)"};}}
.target-card.reg{{background:{"rgba(96,165,250,0.06)" if T=="dark" else "rgba(37,99,235,0.06)"};border-color:{"rgba(96,165,250,0.35)" if T=="dark" else "rgba(37,99,235,0.35)"};}}
.target-card .tc-icon{{font-size:2.5rem;}}
.target-card .tc-label{{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;color:{TEXT3};font-weight:700;}}
.target-card .tc-type{{font-size:1.4rem;font-weight:900;margin:.1rem 0;}}
.target-card.clf .tc-type{{color:{ACCENT1};}}
.target-card.reg .tc-type{{color:{ACCENT2};}}
.target-card .tc-meta{{font-size:.82rem;color:{TEXT2};}}
.step-timeline{{display:flex;flex-direction:column;}}
.step-item{{display:flex;gap:1rem;align-items:flex-start;padding:.75rem 0;position:relative;}}
.step-item:not(:last-child)::before{{content:'';position:absolute;left:17px;top:2.5rem;bottom:-.5rem;width:2px;background:{BORDER};}}
.step-item.done::before{{background:{ACCENT1};}}
.step-dot{{width:34px;height:34px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:.85rem;font-weight:700;border:2px solid {BORDER};background:{BG3};color:{TEXT3};}}
.step-dot.done{{border-color:{ACCENT1};background:{"rgba(74,222,128,0.15)" if T=="dark" else "rgba(124,58,237,0.12)"};color:{ACCENT1};}}
.step-dot.active{{border-color:{ACCENT2};background:{"rgba(96,165,250,0.15)" if T=="dark" else "rgba(37,99,235,0.12)"};color:{ACCENT2};animation:pulse 1.5s infinite;}}
@keyframes pulse{{0%,100%{{box-shadow:0 0 0 0 {"rgba(96,165,250,0.4)" if T=="dark" else "rgba(37,99,235,0.4)"}}}50%{{box-shadow:0 0 0 8px transparent}}}}
.step-label{{font-size:.9rem;font-weight:600;color:{TEXT1};padding-top:.4rem;}}
.step-sub{{font-size:.77rem;color:{TEXT3};}}
.trophy-banner{{border-radius:20px;padding:2rem 2.5rem;background:{"linear-gradient(135deg,rgba(74,222,128,0.08),rgba(96,165,250,0.05))" if T=="dark" else "linear-gradient(135deg,#1e0a3c,#4c1d95)"};border:1px solid {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(167,139,250,0.4)"};display:flex;align-items:center;gap:1.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}}
.trophy-icon{{font-size:3.5rem;flex-shrink:0;}}
.trophy-text h2{{margin:0;font-size:1.6rem;font-weight:900;color:{"#f9fafb" if T=="dark" else "#f5f3ff"};}}
.trophy-text p{{margin:.25rem 0 0;font-size:.9rem;color:{"#9ca3af" if T=="dark" else "#c4b5fd"};}}
.trophy-score{{margin-left:auto;text-align:right;flex-shrink:0;padding:.75rem 1.5rem;background:{"rgba(74,222,128,0.12)" if T=="dark" else "rgba(167,139,250,0.18)"};border-radius:14px;border:1px solid {"rgba(74,222,128,0.25)" if T=="dark" else "rgba(167,139,250,0.35)"}}}
.trophy-score .ts-label{{font-size:.68rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:{"#6b7280" if T=="dark" else "#c4b5fd"};}}
.trophy-score .ts-value{{font-size:2.2rem;font-weight:900;color:{"#4ade80" if T=="dark" else "#e9d5ff"};font-variant-numeric:tabular-nums;}}
.glow-divider{{height:1px;margin:1.5rem 0;background:{GLOW_DIV};opacity:.4;}}
.insight-chip{{display:inline-flex;align-items:center;gap:.4rem;padding:.3rem .8rem;border-radius:8px;font-size:.8rem;font-weight:600;margin:.25rem;background:{BG3};border:1px solid {BORDER};color:{TEXT2};}}
.stTabs [data-baseweb="tab-list"]{{background:{CARD_BG} !important;border:1px solid {BORDER} !important;border-radius:14px !important;padding:6px !important;gap:4px !important;}}
.stTabs [data-baseweb="tab"]{{border-radius:10px !important;font-weight:600 !important;border:none !important;color:{TEXT2} !important;background:transparent !important;padding:.65rem 1.4rem !important;font-size:.9rem !important;}}
.stTabs [data-baseweb="tab"]:hover{{background:{BG3} !important;color:{TEXT1} !important;}}
.stTabs [aria-selected="true"]{{background:{TAB_SEL} !important;color:#fff !important;box-shadow:0 4px 14px {BTN_GLOW} !important;}}
.stSelectbox>div>div{{background:{BG3} !important;border:1px solid {BORDER} !important;border-radius:10px !important;color:{TEXT1} !important;}}
.stTextInput>div>div>input{{background:{BG3} !important;border:1px solid {BORDER} !important;border-radius:10px !important;color:{TEXT1} !important;}}
.stTextInput>div>div>input:focus{{border-color:{ACCENT1} !important;box-shadow:0 0 0 3px {"rgba(74,222,128,0.15)" if T=="dark" else "rgba(124,58,237,0.15)"} !important;}}
.stButton>button{{background:{BTN_BG} !important;color:#fff !important;border:none !important;padding:.8rem 1.75rem !important;font-weight:700 !important;font-size:.95rem !important;border-radius:12px !important;box-shadow:0 4px 16px {BTN_GLOW} !important;letter-spacing:.02em !important;}}
.stButton>button:hover{{transform:translateY(-2px) !important;box-shadow:0 8px 24px {BTN_GLOW} !important;filter:brightness(1.08) !important;}}
.stDownloadButton>button{{background:{BG3} !important;color:{TEXT1} !important;border:1px solid {BORDER} !important;border-radius:12px !important;font-weight:600 !important;}}
div[data-testid="column"]:nth-child(2) .stButton>button{{background:{BG2 if T=="dark" else "#ffffff"} !important;color:{TEXT1} !important;border:1px solid {BORDER} !important;box-shadow:{"0 2px 8px rgba(0,0,0,0.4)" if T=="dark" else "0 2px 8px rgba(0,0,0,0.12)"} !important;font-weight:600 !important;}}
div[data-testid="column"]:nth-child(3) .stButton>button{{background:{BG2 if T=="dark" else "#ffffff"} !important;color:{TEXT1} !important;border:1px solid {BORDER} !important;font-weight:600 !important;}}
div[data-testid="column"]:nth-child(3) .stButton>button:hover{{border-color:{ACCENTR} !important;color:{ACCENTR} !important;}}
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{BG};}}
::-webkit-scrollbar-thumb{{background:{BG4};border-radius:3px;}}
::-webkit-scrollbar-thumb:hover{{background:{ACCENT1};}}
.dataframe{{font-family:'JetBrains Mono',monospace !important;font-size:.82rem !important;}}
.stAlert{{border-radius:12px !important;border-left-width:4px !important;background:{CARD_BG} !important;}}
@keyframes slideUp{{from{{opacity:0;transform:translateY(18px)}}to{{opacity:1;transform:none}}}}
.slide-up{{animation:slideUp .45s ease-out both;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def detect_problem_type(s):
    if s.dtype == "object" or str(s.dtype) == "category": return "classification"
    if s.dtype == "bool": return "classification"
    u, n = s.nunique(), len(s)
    if u <= 10 and pd.api.types.is_integer_dtype(s): return "classification"
    if u / n < 0.05 and u <= 20: return "classification"
    return "regression"

def fmt_time(s):
    return f"{int(s//60)}m {int(s%60)}s" if s >= 60 else f"{s:.1f}s"

def chart_layout(**kwargs):
    base = dict(template=CHART_TEMPLATE, paper_bgcolor=CHART_PAPER, plot_bgcolor=CHART_PAPER,
                font=dict(family="Inter", color=CHART_FONT, size=11),
                margin=dict(t=44, b=20, l=20, r=20), title_font=dict(size=13, color=TEXT1))
    base.update(kwargs)
    return base

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="vibe-header slide-up">
  <div class="vibe-header-content">
    <h1>⚡ DataForge ML Studio</h1>
    <p class="tagline">Drop your data. We handle the rest — AutoML that actually vibes.</p>
    <div class="pill-row">
      <span class="pill pill-green">✦ AutoML</span>
      <span class="pill pill-blue">⚡ 15+ Algorithms</span>
      <span class="pill pill-purple">◈ Smart Detect</span>
      <span class="pill pill-green">↗ Production Ready</span>
      <span class="pill pill-purple">🎁 100% Free</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

_tcol1, _tcol2 = st.columns([10, 1])
with _tcol2:
    if st.button("⬜ White" if T=="dark" else "⬛ Black", key="theme_btn"):
        st.session_state.theme = "light" if T=="dark" else "dark"; st.rerun()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="background:{"rgba(74,222,128,0.08)" if T=="dark" else "#ffffff"};border:1px solid {"rgba(74,222,128,0.30)" if T=="dark" else "#dddddd"};border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem;text-align:center">
      <div style="font-size:1.5rem;margin-bottom:.3rem">🎁</div>
      <div style="font-size:.85rem;font-weight:800;color:{"#4ade80" if T=="dark" else "#7c3aed"}">All Features Free</div>
      <div style="font-size:.72rem;color:{"#9ca3af" if T=="dark" else "#888"};margin-top:.2rem">XGBoost · LightGBM · CatBoost · 10-fold CV · Model Export</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-title">📂 Data Source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"], label_visibility="collapsed")
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            if st.session_state.get("_last_uploaded") != uploaded.name:
                st.session_state.data = df_up
                st.session_state.dataset_name = uploaded.name
                st.session_state.results = None
                st.session_state.best_model = None
                st.session_state["_last_uploaded"] = uploaded.name
                st.session_state.pop("sample_hint", None)
                st.rerun()
        except Exception as e:
            st.error(str(e))

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-title">🎯 Sample Datasets</div>', unsafe_allow_html=True)
    sample = st.selectbox("Pick one", ["— choose —","🚢 Titanic","💎 Diamonds","🌸 Iris"], label_visibility="collapsed", key="sample_dataset_select", index=0)
    if sample != "— choose —":
        if st.button("Load Sample →", key="load_sample"):
            try:
                urls = {
                    "🚢 Titanic":  ("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv","Survived"),
                    "💎 Diamonds": ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv","price"),
                    "🌸 Iris":     ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv","species"),
                }
                url, hint = urls[sample]
                df_s = pd.read_csv(url)
                st.session_state.data = df_s
                st.session_state.dataset_name = sample
                st.session_state.results = None
                st.session_state.best_model = None
                st.session_state["sample_hint"] = hint
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if st.session_state.get("sample_hint"):
        st.info(f"💡 Target: **{st.session_state['sample_hint']}**")

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

    if st.session_state.data is not None:
        df_sb = st.session_state.data
        null_pct_sb = round(df_sb.isnull().sum().sum() / df_sb.size * 100, 2)
        hs = round(max(0, 100 - null_pct_sb * 2 - df_sb.duplicated().sum() / len(df_sb) * 300), 1)
        hc = ACCENT1 if hs > 80 else ACCENTY if hs > 50 else ACCENTR
        st.markdown(f"""
        <div class="sidebar-section">
          <div class="sidebar-title">📊 Dataset Health</div>
          <div style="font-size:2.2rem;font-weight:900;color:{hc}">{hs}<span style="font-size:1rem;color:{TEXT3}">/100</span></div>
          <div style="font-size:.75rem;color:{TEXT3};margin-top:.3rem">{df_sb.isnull().sum().sum()} nulls · {df_sb.duplicated().sum()} duplicates</div>
          <div style="height:6px;background:{'#1c1c1c' if T=='dark' else '#e0e0e0'};border-radius:3px;margin-top:.75rem;overflow:hidden">
            <div style="height:100%;width:{hs}%;background:linear-gradient(90deg,{hc},{hc}99);border-radius:3px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-title">⚡ Stack</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-section"><span class="insight-chip">PyCaret</span><span class="insight-chip">Plotly</span><span class="insight-chip">Pandas</span><span class="insight-chip">Streamlit</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────
if st.session_state.data is not None:
    df = st.session_state.data
    null_pct = round(df.isnull().sum().sum() / df.size * 100, 2)
    dup_cnt  = df.duplicated().sum()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊  Data Explorer", "🧬  EDA & Insights", "⚙️  Train Model",
        "🏆  Results", "📜  History", "📓  Notebook Builder"
    ])

    # ═══════════════════════════
    # TAB 1 — DATA EXPLORER
    # ═══════════════════════════
    with tab1:
        st.markdown(f"""
        <div class="stat-grid slide-up">
          <div class="stat-card {'good' if null_pct<=5 else 'warn' if null_pct<=15 else 'danger'}">
            <div class="bar"></div><div class="label">Total Rows</div>
            <div class="value">{len(df):,}</div><div class="sub">records</div>
          </div>
          <div class="stat-card">
            <div class="bar"></div><div class="label">Columns</div>
            <div class="value">{len(df.columns)}</div><div class="sub">features</div>
          </div>
          <div class="stat-card">
            <div class="bar"></div><div class="label">Numerical</div>
            <div class="value">{len(num_cols)}</div><div class="sub">numeric cols</div>
          </div>
          <div class="stat-card">
            <div class="bar"></div><div class="label">Categorical</div>
            <div class="value">{len(cat_cols)}</div><div class="sub">text cols</div>
          </div>
          <div class="stat-card {'good' if null_pct==0 else 'warn' if null_pct<=10 else 'danger'}">
            <div class="bar"></div><div class="label">Missing</div>
            <div class="value">{null_pct}%</div><div class="sub">{df.isnull().sum().sum()} cells</div>
          </div>
        </div>""", unsafe_allow_html=True)

        if dup_cnt > 0:
            st.warning(f"⚠️ **{dup_cnt}** duplicate rows found. PyCaret will handle automatically.")

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">⚡</div><h3>Quick Actions</h3></div>""", unsafe_allow_html=True)
        qa1, qa2, qa3, qa4 = st.columns(4)
        with qa1:
            if st.button("🗑️ Drop Duplicates", key="drop_dups"):
                before = len(df)
                st.session_state.data = df.drop_duplicates().reset_index(drop=True)
                st.success(f"Removed {before - len(st.session_state.data)} duplicates!"); st.rerun()
        with qa2:
            if st.button("🧹 Drop All-Null Cols", key="drop_null_cols"):
                before = len(df.columns)
                st.session_state.data = df.dropna(axis=1, how='all')
                st.success(f"Removed {before - len(st.session_state.data.columns)} empty columns!"); st.rerun()
        with qa3:
            if st.button("📊 Show Data Types", key="show_dtypes"):
                st.dataframe(df.dtypes.reset_index().rename(columns={"index":"Column",0:"Type"}))
        with qa4:
            st.download_button("📥 Download CSV", df.to_csv(index=False),
                               f"dataset_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", "text/csv")

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">🔍</div><h3>Data Preview</h3></div>""", unsafe_allow_html=True)
        search = st.text_input("Filter columns (comma-separated)", placeholder="e.g. Age, Sex, Survived", label_visibility="collapsed")
        show_rows = st.slider("Rows to show", 5, 100, 20, key="preview_rows")
        if search.strip():
            cols_f = [c.strip() for c in search.split(",") if c.strip() in df.columns]
            st.dataframe((df[cols_f] if cols_f else df).head(show_rows))
        else:
            st.dataframe(df.head(show_rows))

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📋</div><h3>Column Details</h3></div>""", unsafe_allow_html=True)
            ci = pd.DataFrame({"Column":df.columns,"Type":df.dtypes.astype(str).values,
                                "Non-Null":df.count().values,
                                "Null %":((df.isnull().sum()/len(df))*100).round(1).astype(str)+"%",
                                "Unique":df.nunique().values})
            st.dataframe(ci.astype(str), height=300)
        with c2:
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📊</div><h3>Statistical Summary</h3></div>""", unsafe_allow_html=True)
            st.dataframe(df.describe().round(3), height=300)

    # ═══════════════════════════
    # TAB 2 — EDA
    # ═══════════════════════════
    with tab2:
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">🧬</div><h3>Exploratory Data Analysis</h3></div>""", unsafe_allow_html=True)

        if not num_cols:
            st.info("No numerical columns found.")
        else:
            st.markdown("#### 📈 Distribution Explorer")
            dv1, dv2 = st.columns([3, 2])
            with dv1:
                col_pick = st.selectbox("Select column", num_cols + cat_cols, key="eda_col")
            with dv2:
                chart_type = st.selectbox("Chart type", ["Histogram","Box","Violin"] if col_pick in num_cols else ["Bar Chart"], key="chart_type")

            cv1, cv2 = st.columns([3, 2])
            with cv1:
                if col_pick in num_cols:
                    if chart_type == "Histogram":
                        fig = px.histogram(df, x=col_pick, nbins=40, color_discrete_sequence=[ACCENT1], template=CHART_TEMPLATE, title=f"Distribution · {col_pick}")
                    elif chart_type == "Box":
                        fig = px.box(df, y=col_pick, color_discrete_sequence=[ACCENT1], template=CHART_TEMPLATE, title=f"Box Plot · {col_pick}")
                    else:
                        fig = px.violin(df, y=col_pick, color_discrete_sequence=[ACCENT1], box=True, template=CHART_TEMPLATE, title=f"Violin · {col_pick}")
                else:
                    vc = df[col_pick].value_counts().head(15)
                    fig = px.bar(x=vc.index, y=vc.values, color=vc.values, color_continuous_scale=[ACCENT2, ACCENT1], template=CHART_TEMPLATE, title=f"Top values · {col_pick}")
                    fig.update_layout(showlegend=False, coloraxis_showscale=False)
                fig.update_layout(**chart_layout(height=340))
                st.plotly_chart(fig, width="stretch")

            with cv2:
                s = df[col_pick]
                rows = {"Count":f"{s.count():,}","Missing":f"{s.isnull().sum()} ({s.isnull().mean()*100:.1f}%)","Unique":f"{s.nunique():,}"}
                if col_pick in num_cols:
                    rows.update({"Mean":f"{s.mean():.4f}","Std":f"{s.std():.4f}","Min":f"{s.min():.4f}","Median":f"{s.median():.4f}","Max":f"{s.max():.4f}","Skew":f"{s.skew():.3f}"})
                st.markdown(f'<div class="sidebar-section" style="margin-top:2.2rem">', unsafe_allow_html=True)
                for k, v in rows.items():
                    st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid {BORDER}"><span style="font-size:.78rem;color:{TEXT3};font-weight:600">{k}</span><span style="font-size:.82rem;color:{TEXT1};font-family:'JetBrains Mono',monospace">{v}</span></div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            if len(num_cols) >= 2:
                st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown("#### 🔥 Correlation Heatmap")
                corr = df[num_cols[:15]].corr().round(2)
                fig_h = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.index,
                    colorscale=[[0,ACCENTR],[.5,BG2],[1,ACCENT1]],
                    zmid=0, text=corr.values.round(2), texttemplate="%{text}",
                    textfont=dict(size=9, family="JetBrains Mono")))
                fig_h.update_layout(**chart_layout(height=460))
                st.plotly_chart(fig_h, width="stretch")

    # ═══════════════════════════
    # TAB 3 — TRAIN MODEL
    # ═══════════════════════════
    with tab3:
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">⚙️</div><h3>Training Configuration</h3></div>""", unsafe_allow_html=True)

        tc1, tc2 = st.columns([3, 1])
        with tc1:
            target_col = st.selectbox("🎯 Select Target Column", df.columns.tolist())
        with tc2:
            st.markdown("<br>", unsafe_allow_html=True)
            row_color = ACCENT1 if len(df) <= MAX_ROWS_WARNING else ACCENTY if len(df) <= MAX_ROWS_TRAINING else ACCENTR
            st.markdown(
                f'<div style="padding:.6rem 1rem;background:{BG3};border:1px solid {row_color}44;border-radius:10px;text-align:center">'
                f'<div style="font-size:.65rem;color:{TEXT3};text-transform:uppercase;font-weight:700">Dataset Size</div>'
                f'<div style="font-size:1.1rem;font-weight:900;color:{row_color}">{len(df):,}</div>'
                f'<div style="font-size:.62rem;color:{TEXT3}">rows</div></div>',
                unsafe_allow_html=True
            )

        if target_col:
            ts = df[target_col]
            ptype = detect_problem_type(ts)
            uniq = ts.nunique()
            st.session_state.problem_type = ptype

            card_cls = "clf" if ptype == "classification" else "reg"
            icon = "🎯" if ptype == "classification" else "📈"
            type_lbl = "Classification" if ptype == "classification" else "Regression"
            st.markdown(f"""
            <div class="target-card {card_cls} slide-up">
              <div class="tc-icon">{icon}</div>
              <div>
                <div class="tc-label">Problem Type Detected</div>
                <div class="tc-type">{type_lbl}</div>
                <div class="tc-meta">Target: <code>{target_col}</code> &nbsp;·&nbsp; {uniq} unique values &nbsp;·&nbsp; Auto cross-validation enabled</div>
              </div>
            </div>""", unsafe_allow_html=True)

            if len(df) > MAX_ROWS_TRAINING:
                st.error(
                    f"🚨 **Dataset {len(df):,} rows** — too large for free hosting.  \n"
                    f"Training will auto-sample **{MAX_ROWS_TRAINING:,} rows** (stratified)."
                )
            elif len(df) > MAX_ROWS_WARNING:
                st.warning(
                    f"⚠️ **{len(df):,} rows** — thoda bada hai. Training chalegi lekin "
                    f"agar crash ho toh {MAX_ROWS_WARNING:,} rows tak chota karo."
                )

            available_models = ALL_CLF_MODELS if ptype == "classification" else ALL_REG_MODELS

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(74,222,128,0.12),rgba(96,165,250,0.08));
                        border:1.5px solid rgba(74,222,128,0.4);border-radius:14px;
                        padding:.9rem 1.25rem;margin:.75rem 0;display:flex;align-items:center;gap:1rem">
              <div style="font-size:2rem">🎁</div>
              <div>
                <div style="font-size:.9rem;font-weight:900;color:#4ade80;letter-spacing:.02em">
                  ALL FEATURES UNLOCKED — 100% FREE
                </div>
                <div style="font-size:.72rem;color:#9ca3af;margin-top:.15rem">
                  XGBoost ✅ &nbsp; LightGBM ✅ &nbsp; CatBoost ✅ &nbsp; 10-fold CV ✅ &nbsp; Unlimited Training ✅ &nbsp; Model Export ✅
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:{"rgba(255,255,255,0.03)" if T=="dark" else "rgba(0,0,0,0.03)"};border:1px solid {BORDER};border-radius:12px;padding:1rem 1.25rem;margin:.75rem 0">
              <span class="insight-chip">🤖 {len(available_models)} algorithms available</span>
              <span class="insight-chip">📦 XGBoost ✅ LightGBM ✅ CatBoost ✅</span>
              <span class="insight-chip">💾 Max {MAX_ROWS_TRAINING:,} rows (auto-sample)</span>
              <span class="insight-chip">💡 Free forever</span>
            </div>""", unsafe_allow_html=True)

            with st.expander("⚙️ Advanced Configuration", expanded=False):
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    train_size = st.slider("Training Split", 0.5, 0.9, 0.8, 0.05)
                with ac2:
                    recommended_fold = min(3, 10) if len(df) > MAX_ROWS_WARNING else 5
                    fold = st.slider(
                        "CV Folds (max 10)",
                        min_value=2, max_value=10,
                        value=min(recommended_fold, 10),
                        help="Higher folds = more accurate but slower. 5 is usually optimal."
                    )
                with ac3:
                    max_models_slider = st.slider(
                        f"Max Models ({len(available_models)} available)",
                        min_value=2,
                        max_value=len(available_models),
                        value=len(available_models),
                    )
                ac4, ac5 = st.columns(2)
                with ac4:
                    normalize = st.checkbox("Normalize Features", value=True)
                with ac5:
                    remove_out = st.checkbox("Remove Outliers", value=False)
                st.markdown(
                    f'<div style="background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.30);'
                    f'border-radius:8px;padding:.6rem .9rem;font-size:.78rem;color:{ACCENTY}">'
                    f'💡 <b>Memory Tip:</b> Fewer folds = less RAM. Bade datasets ke liye 2-3 folds use karo.</div>',
                    unsafe_allow_html=True
                )

            st.session_state.cv_fold = fold

            st.markdown("<br>", unsafe_allow_html=True)
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                train_clicked = st.button("🚀 Launch Training", key="train_btn")
            with col_btn2:
                if st.session_state.results is not None:
                    if st.button("🔄 Reset Results", key="reset_btn"):
                        st.session_state.results = None
                        st.session_state.best_model = None
                        st.session_state.training_time = None
                        force_gc()
                        st.rerun()

            if train_clicked:
                progress_bar = st.progress(0)
                status_box   = st.empty()
                warn_box     = st.empty()
                timeline_box = st.empty()

                steps_labels = [
                    "📦 Data Sampling & Validation",
                    "⚙️ PyCaret Environment Setup",
                    "🤖 Model Comparison (memory-safe)",
                    "🏆 Best Model Selection",
                    "💾 Saving Artifact",
                ]

                def render_steps(done_count):
                    html = '<div class="step-timeline">'
                    for i, lbl in enumerate(steps_labels):
                        cls    = "done"   if i < done_count else ("active" if i == done_count else "")
                        icon_s = "✓"      if i < done_count else ("◉"      if i == done_count else str(i+1))
                        html += (
                            f'<div class="step-item {cls}">'
                            f'<div class="step-dot {cls}">{icon_s}</div>'
                            f'<div><div class="step-label">{lbl}</div></div>'
                            f'</div>'
                        )
                    return html + "</div>"

                timeline_box.markdown(render_steps(0), unsafe_allow_html=True)
                progress_bar.progress(5)
                status_box.info("🚀 Training is starting...")

                try:
                    best, results, elapsed, warn_msgs, trained_rows = run_memory_safe_training(
                        df           = df,
                        target_col   = target_col,
                        problem_type = ptype,
                        train_size   = train_size,
                        fold         = fold,
                        normalize    = normalize,
                        remove_out   = remove_out,
                        max_models   = max_models_slider,
                    )

                    progress_bar.progress(100)
                    timeline_box.markdown(render_steps(len(steps_labels)), unsafe_allow_html=True)

                    st.session_state.best_model    = best
                    st.session_state.results       = results
                    st.session_state.training_time = elapsed

                    for w in warn_msgs:
                        warn_box.warning(w)

                    if "training_history" not in st.session_state:
                        st.session_state.training_history = []
                    try:
                        model_col_r = "Model" if "Model" in results.columns else results.columns[0]
                        num_res_r   = results.select_dtypes(include=[np.number]).columns
                        bm_name_r   = str(results.iloc[0][model_col_r])
                        bm_score_r  = float(results.iloc[0][num_res_r[0]]) if len(num_res_r) else 0.0
                        st.session_state.training_history.append({
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "dataset": str(st.session_state.dataset_name or "Uploaded CSV"),
                            "problem_type": str(ptype),
                            "best_model": bm_name_r,
                            "score": round(bm_score_r, 4),
                            "rows": trained_rows,
                            "cols": len(df.columns)
                        })
                    except Exception:
                        pass

                    status_box.success(
                        f"✅ Training complete in **{fmt_time(elapsed)}** "
                        f"({trained_rows:,} rows) — 🏆 Check the Results tab!"
                    )
                    st.balloons()

                except MemoryError as me:
                    progress_bar.progress(0)
                    timeline_box.empty()
                    status_box.error(
                        f"💥 **Memory Overflow!**  \n{str(me)}  \n\n"
                        f"**Quick fixes:**  \n"
                        f"- Dataset ko {MAX_ROWS_TRAINING:,} rows se kam karo  \n"
                        f"- Page refresh karo (Ctrl+R) aur dobara try karo  \n"
                        f"- CV Folds ko 2 par set karo"
                    )
                except Exception as e:
                    progress_bar.progress(0)
                    timeline_box.empty()
                    err_str = str(e)
                    if any(kw in err_str.lower() for kw in ["memory","killed","oom","cannot allocate"]):
                        status_box.error("💥 **Memory Crash!** Dataset chota karo aur page refresh karo.")
                    else:
                        status_box.error(f"❌ Training failed: {err_str}")

    # ═══════════════════════════
    # TAB 4 — RESULTS
    # ═══════════════════════════
    with tab4:
        if st.session_state.results is None:
            st.markdown(f"""
            <div style="text-align:center;padding:5rem 2rem">
              <div style="font-size:5rem;margin-bottom:1rem;opacity:.6">🏆</div>
              <div style="font-size:1.4rem;font-weight:800;color:{TEXT1};margin-bottom:.5rem">No results yet</div>
              <div style="color:{TEXT2}">Train a model in the ⚙️ Train Model tab first</div>
            </div>""", unsafe_allow_html=True)
        else:
            res_df = st.session_state.results
            model_col   = "Model" if "Model" in res_df.columns else res_df.columns[0]
            num_res     = res_df.select_dtypes(include=[np.number]).columns.tolist()
            best_name   = res_df.iloc[0][model_col]
            metric_name = num_res[0] if num_res else "Score"
            top_score   = res_df.iloc[0][metric_name] if num_res else 0
            folds_used  = st.session_state.cv_fold or 5

            st.markdown(f"""
            <div class="trophy-banner slide-up">
              <div class="trophy-icon">🏆</div>
              <div class="trophy-text">
                <h2>{best_name}</h2>
                <p>Best model via {folds_used}-fold cross-validation · Production ready</p>
              </div>
              <div class="trophy-score">
                <div class="ts-label">{metric_name}</div>
                <div class="ts-value">{top_score:.4f}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # ── Download buttons: 3 columns ──
            ex1, ex2, ex3 = st.columns(3)
            with ex1:
                st.download_button("📥 Export Results CSV", res_df.to_csv(index=False),
                                   f"results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            with ex2:
                if os.path.exists("best_model.pkl"):
                    with open("best_model.pkl", "rb") as pkl_f:
                        pkl_bytes = pkl_f.read()
                    st.download_button(
                        "📦 Download Model (.pkl)",
                        data=pkl_bytes,
                        file_name=f"best_model_{best_name.replace(' ','_')}.pkl",
                        mime="application/octet-stream",
                        help="Trained model file — load with PyCaret's load_model()"
                    )
                else:
                    st.info("💾 Model file generates after training. Re-train to get it.")
            with ex3:
                try:
                    nb_bytes = generate_notebook(
                        df              = df,
                        target_col      = target_col,
                        problem_type    = st.session_state.problem_type,
                        results_df      = res_df,
                        best_model_name = best_name,
                        top_score       = top_score,
                        metric_name     = metric_name,
                        dataset_name    = str(st.session_state.dataset_name or "dataset"),
                        training_time   = st.session_state.training_time or 0,
                        fold            = st.session_state.cv_fold or 5,
                        normalize       = True,
                        train_size      = 0.8,
                        eda_selections  = st.session_state.get("nb_eda"),
                        pre_selections  = st.session_state.get("nb_pre"),
                        train_selections= st.session_state.get("nb_train"),
                        export_selections=st.session_state.get("nb_export"),
                    )
                    safe_name = str(st.session_state.dataset_name or "dataset").replace(" ","_").replace(".","_")
                    st.download_button(
                        label     = "📓 Download Notebook (.ipynb)",
                        data      = nb_bytes,
                        file_name = f"dataforge_{safe_name}_{best_name.replace(' ','_')}.ipynb",
                        mime      = "application/x-ipynb+json",
                        help      = "Complete Jupyter Notebook — EDA + Training + Results + Model Export"
                    )
                except Exception as nb_err:
                    st.error(f"Notebook generation failed: {nb_err}")

            st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📋</div><h3>All Models Ranked</h3></div>""", unsafe_allow_html=True)
            styled = (res_df.style
                      .background_gradient(cmap="RdYlGn", subset=num_res)
                      .format({c: "{:.4f}" for c in num_res})
                      .set_properties(**{"font-family":"JetBrains Mono,monospace","font-size":"12px"}))
            st.dataframe(styled, height=360)

            st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
            ch1, ch2 = st.columns(2)
            with ch1:
                top6   = res_df.head(6)
                colors = [ACCENT1 if i == 0 else BG3 for i in range(len(top6))]
                fig_b  = go.Figure(go.Bar(x=top6[metric_name], y=top6[model_col], orientation="h",
                    marker_color=colors, text=top6[metric_name].round(4), textposition="inside",
                    textfont=dict(size=10, color="white")))
                fig_b.update_layout(**chart_layout(height=360, title=f"Top Models · {metric_name}", yaxis=dict(autorange="reversed")))
                st.plotly_chart(fig_b, width="stretch")
            with ch2:
                rc = num_res[:6]
                bv = res_df.iloc[0][rc]
                mi, ma = bv.min(), bv.max()
                nv = (bv - mi) / (ma - mi + 1e-9)
                fig_r = go.Figure(go.Scatterpolar(
                    r=list(nv.values)+[nv.values[0]], theta=list(nv.index)+[nv.index[0]],
                    fill="toself", fillcolor="rgba(74,222,128,0.18)",
                    line=dict(color=ACCENT1, width=2.5), marker=dict(size=6, color=ACCENT1)))
                fig_r.update_layout(**chart_layout(height=360, showlegend=False, title="Best Model · Metrics Radar",
                    polar=dict(bgcolor=CHART_PAPER, radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER),
                               angularaxis=dict(gridcolor=BORDER))))
                st.plotly_chart(fig_r, width="stretch")

    # ═══════════════════════════
    # TAB 5 — HISTORY (session only)
    # ═══════════════════════════
    with tab5:
        training_log = st.session_state.get("training_history", [])

        st.markdown(f"""<div class="section-head"><div class="icon-wrap">🗂️</div><h3>This Session's Training History</h3></div>""", unsafe_allow_html=True)
        st.caption("History is saved for this browser session. Refresh page se history reset hoti hai.")

        if not training_log:
            st.markdown(f"""
            <div style="text-align:center;padding:3rem 1rem;background:{CARD_BG};border:1px solid {BORDER};border-radius:20px">
              <div style="font-size:4rem;margin-bottom:.75rem;opacity:.35">🗂️</div>
              <div style="font-size:1.1rem;font-weight:700;color:{TEXT1}">No projects yet</div>
              <div style="color:{TEXT2};font-size:.875rem">Upload a dataset and train your first model!</div>
            </div>""", unsafe_allow_html=True)
        else:
            for t in training_log[::-1]:
                pt = t.get("problem_type","—")
                pt_color = ACCENT1 if pt == "classification" else ACCENT2
                st.markdown(f"""
                <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;padding:1.25rem;margin-bottom:.75rem;display:flex;align-items:center;gap:1rem;position:relative;overflow:hidden">
                  <div style="position:absolute;top:0;left:0;bottom:0;width:3px;background:{pt_color}"></div>
                  <div style="font-size:1.5rem;margin-left:.5rem">{"🎯" if pt=="classification" else "📈"}</div>
                  <div style="flex:1">
                    <div style="font-size:.9rem;font-weight:700;color:{TEXT1}">{t.get("dataset","?")}</div>
                    <div style="font-size:.75rem;color:{TEXT3};margin-top:.15rem">{t.get("best_model","?")} · {t.get("rows",0):,} rows · {t.get("time","")[:16]}</div>
                  </div>
                  <div style="text-align:right">
                    <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;color:{TEXT3}">Score</div>
                    <div style="font-size:1.3rem;font-weight:900;color:{pt_color};font-family:'JetBrains Mono',monospace">{t.get("score",0):.4f}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            tlog_df = pd.DataFrame(training_log)
            st.download_button("📥 Export History CSV", tlog_df.to_csv(index=False),
                               "training_history.csv", "text/csv")

    # ═══════════════════════════════════════════
    # TAB 6 — NOTEBOOK BUILDER
    # ═══════════════════════════════════════════
    with tab6:

        # ── Session state init for notebook builder ──
        if "nb_step" not in st.session_state:
            st.session_state.nb_step = 1
        if "nb_ptype" not in st.session_state:
            st.session_state.nb_ptype = st.session_state.problem_type or "classification"
        if "nb_eda" not in st.session_state:
            st.session_state.nb_eda = {
                "distributions": True, "correlation": True, "missing": True,
                "target_dist": True, "boxplots": False, "scatter_matrix": False,
                "cat_bars": False, "outlier_plot": False,
            }
        if "nb_pre" not in st.session_state:
            st.session_state.nb_pre = {
                "drop_dups": True, "handle_missing": True,
                "normalize": True, "remove_outliers": False,
            }
        if "nb_train" not in st.session_state:
            st.session_state.nb_train = {
                "model_table": True, "bar_chart": True,
                "radar_chart": True, "feature_importance": False,
            }
        if "nb_export" not in st.session_state:
            st.session_state.nb_export = {
                "save_model": True, "load_predict": True, "summary_table": True,
            }

        step = st.session_state.nb_step

        # ── Step indicator CSS ──
        st.markdown(f"""
        <style>
        .nb-step-row{{display:flex;gap:14px;margin-bottom:20px;align-items:flex-start;}}
        .nb-dot{{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.8rem;font-weight:700;flex-shrink:0;margin-top:2px;}}
        .nb-dot-done{{background:{"rgba(74,222,128,0.15)" if T=="dark" else "rgba(74,222,128,0.20)"};color:#4ade80;border:2px solid #4ade80;}}
        .nb-dot-active{{background:{"rgba(96,165,250,0.15)" if T=="dark" else "rgba(96,165,250,0.15)"};color:#60a5fa;border:2px solid #60a5fa;animation:nb-pulse 1.5s infinite;}}
        .nb-dot-idle{{background:{BG3};color:{TEXT3};border:2px solid {BORDER};}}
        @keyframes nb-pulse{{0%,100%{{box-shadow:0 0 0 0 rgba(96,165,250,0.4)}}50%{{box-shadow:0 0 0 7px transparent}}}}
        .nb-line{{width:2px;min-height:16px;margin:3px auto;}}
        .nb-line-done{{background:#4ade80;}}
        .nb-line-idle{{background:{BORDER};}}
        .nb-step-title{{font-size:.95rem;font-weight:700;color:{TEXT1};padding-top:5px;}}
        .nb-step-title-idle{{font-size:.95rem;font-weight:600;color:{TEXT3};padding-top:5px;}}
        .nb-chip-grid{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;}}
        .nb-chip-on{{display:inline-flex;align-items:center;gap:5px;padding:5px 13px;border-radius:8px;font-size:.78rem;font-weight:600;background:{"rgba(74,222,128,0.12)" if T=="dark" else "rgba(74,222,128,0.15)"};color:#4ade80;border:1px solid rgba(74,222,128,0.4);cursor:pointer;}}
        .nb-chip-off{{display:inline-flex;align-items:center;gap:5px;padding:5px 13px;border-radius:8px;font-size:.78rem;font-weight:600;background:{BG3};color:{TEXT3};border:1px solid {BORDER};cursor:pointer;}}
        .nb-toggle-row{{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid {BORDER};}}
        .nb-toggle-row:last-child{{border-bottom:none;}}
        .nb-tgl-label{{font-size:.85rem;font-weight:600;color:{TEXT1};}}
        .nb-tgl-sub{{font-size:.72rem;color:{TEXT3};margin-top:2px;}}
        .nb-card{{background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:1rem 1.25rem;margin-top:10px;}}
        </style>
        """, unsafe_allow_html=True)

        # ── Header ──
        st.markdown(f"""
        <div class="section-head"><div class="icon-wrap">📓</div><h3>Notebook Builder</h3></div>
        <div style="font-size:.85rem;color:{TEXT2};margin-bottom:1.5rem">
          Step-by-step select karo kya kya notebook mein add karna hai — phir ek click mein complete <b>.ipynb</b> download hoga.
        </div>
        """, unsafe_allow_html=True)

        # ────────────────────────────────
        # STEP 1 — Problem Type
        # ────────────────────────────────
        def dot(n):
            if n < step: return f'<div class="nb-dot nb-dot-done">✓</div>'
            elif n == step: return f'<div class="nb-dot nb-dot-active">{n}</div>'
            else: return f'<div class="nb-dot nb-dot-idle">{n}</div>'

        def line(n):
            cls = "nb-line-done" if n < step else "nb-line-idle"
            return f'<div class="nb-line {cls}" style="height:12px"></div>'

        # Step 1
        with st.container():
            col_dot, col_body = st.columns([1, 12])
            with col_dot:
                st.markdown(f'<div style="display:flex;flex-direction:column;align-items:center">{dot(1)}{line(1)}</div>', unsafe_allow_html=True)
            with col_body:
                if step >= 1:
                    st.markdown(f'<div class="nb-step-title">Problem type select karo</div>', unsafe_allow_html=True)
                    ptype_options = ["Classification", "Regression"]
                    current_ptype = "Classification" if st.session_state.nb_ptype == "classification" else "Regression"
                    chosen = st.radio("Problem type", ptype_options, index=ptype_options.index(current_ptype),
                                      horizontal=True, label_visibility="collapsed", key="nb_ptype_radio")
                    st.session_state.nb_ptype = "classification" if chosen == "Classification" else "regression"
                    if step == 1:
                        if st.button("Continue →", key="nb_next1"):
                            st.session_state.nb_step = 2; st.rerun()
                else:
                    st.markdown(f'<div class="nb-step-title-idle">Problem type select karo</div>', unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # Step 2 — EDA Charts
        with st.container():
            col_dot, col_body = st.columns([1, 12])
            with col_dot:
                st.markdown(f'<div style="display:flex;flex-direction:column;align-items:center">{dot(2)}{line(2)}</div>', unsafe_allow_html=True)
            with col_body:
                if step >= 2:
                    st.markdown(f'<div class="nb-step-title">EDA — charts select karo</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:.8rem;color:{TEXT2};margin:.4rem 0 .75rem">Jo charts chahiye notebook mein, woh ON karo:</div>', unsafe_allow_html=True)

                    eda = st.session_state.nb_eda
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        eda["distributions"] = st.checkbox("📊 Distributions histogram", value=eda["distributions"], key="eda_dist")
                        eda["correlation"]   = st.checkbox("🔥 Correlation heatmap",     value=eda["correlation"],   key="eda_corr")
                        eda["missing"]       = st.checkbox("❓ Missing values bar chart", value=eda["missing"],       key="eda_miss")
                        eda["target_dist"]   = st.checkbox("🎯 Target distribution",      value=eda["target_dist"],   key="eda_tgt")
                    with ec2:
                        eda["boxplots"]       = st.checkbox("📦 Box plots",               value=eda["boxplots"],       key="eda_box")
                        eda["scatter_matrix"] = st.checkbox("🔵 Scatter matrix",          value=eda["scatter_matrix"], key="eda_scat")
                        eda["cat_bars"]       = st.checkbox("📋 Category bar charts",     value=eda["cat_bars"],       key="eda_cat")
                        eda["outlier_plot"]   = st.checkbox("⚠️ Outlier detection plot",  value=eda["outlier_plot"],   key="eda_out")
                    st.session_state.nb_eda = eda

                    selected_count = sum(1 for v in eda.values() if v)
                    st.markdown(f'<div style="font-size:.75rem;color:{ACCENT1};margin-top:.5rem">✓ {selected_count} chart(s) selected</div>', unsafe_allow_html=True)

                    if step == 2:
                        nb2c1, nb2c2 = st.columns([1, 4])
                        with nb2c1:
                            if st.button("← Back", key="nb_back2"):
                                st.session_state.nb_step = 1; st.rerun()
                        with nb2c2:
                            if st.button("Continue →", key="nb_next2"):
                                st.session_state.nb_step = 3; st.rerun()
                else:
                    st.markdown(f'<div class="nb-step-title-idle">EDA — charts select karo</div>', unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # Step 3 — Preprocessing
        with st.container():
            col_dot, col_body = st.columns([1, 12])
            with col_dot:
                st.markdown(f'<div style="display:flex;flex-direction:column;align-items:center">{dot(3)}{line(3)}</div>', unsafe_allow_html=True)
            with col_body:
                if step >= 3:
                    st.markdown(f'<div class="nb-step-title">Preprocessing steps</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:.8rem;color:{TEXT2};margin:.4rem 0 .75rem">Notebook mein kaunse preprocessing steps include karne hain:</div>', unsafe_allow_html=True)

                    pre = st.session_state.nb_pre
                    pre["drop_dups"]       = st.checkbox("🗑️ Drop duplicate rows",      value=pre["drop_dups"],       key="pre_dups")
                    pre["handle_missing"]  = st.checkbox("🩹 Handle missing values",     value=pre["handle_missing"],  key="pre_miss")
                    pre["normalize"]       = st.checkbox("⚖️ Normalize features",        value=pre["normalize"],       key="pre_norm")
                    pre["remove_outliers"] = st.checkbox("🚫 Remove outliers (reg only)",value=pre["remove_outliers"], key="pre_out")
                    st.session_state.nb_pre = pre

                    if step == 3:
                        nb3c1, nb3c2 = st.columns([1, 4])
                        with nb3c1:
                            if st.button("← Back", key="nb_back3"):
                                st.session_state.nb_step = 2; st.rerun()
                        with nb3c2:
                            if st.button("Continue →", key="nb_next3"):
                                st.session_state.nb_step = 4; st.rerun()
                else:
                    st.markdown(f'<div class="nb-step-title-idle">Preprocessing steps</div>', unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # Step 4 — Training / Results Charts
        with st.container():
            col_dot, col_body = st.columns([1, 12])
            with col_dot:
                st.markdown(f'<div style="display:flex;flex-direction:column;align-items:center">{dot(4)}{line(4)}</div>', unsafe_allow_html=True)
            with col_body:
                if step >= 4:
                    st.markdown(f'<div class="nb-step-title">Training & results charts</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:.8rem;color:{TEXT2};margin:.4rem 0 .75rem">Results section mein kaunse charts add karne hain:</div>', unsafe_allow_html=True)

                    tr = st.session_state.nb_train
                    tr["model_table"]       = st.checkbox("📋 All models comparison table",  value=tr["model_table"],       key="tr_tbl")
                    tr["bar_chart"]         = st.checkbox("📊 Top models bar chart",         value=tr["bar_chart"],         key="tr_bar")
                    tr["radar_chart"]       = st.checkbox("🕸️ Best model metrics radar",     value=tr["radar_chart"],       key="tr_rad")
                    tr["feature_importance"]= st.checkbox("⭐ Feature importance plot",      value=tr["feature_importance"],key="tr_feat")
                    st.session_state.nb_train = tr

                    if step == 4:
                        nb4c1, nb4c2 = st.columns([1, 4])
                        with nb4c1:
                            if st.button("← Back", key="nb_back4"):
                                st.session_state.nb_step = 3; st.rerun()
                        with nb4c2:
                            if st.button("Continue →", key="nb_next4"):
                                st.session_state.nb_step = 5; st.rerun()
                else:
                    st.markdown(f'<div class="nb-step-title-idle">Training & results charts</div>', unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # Step 5 — Export + Generate
        with st.container():
            col_dot, col_body = st.columns([1, 12])
            with col_dot:
                st.markdown(f'<div style="display:flex;flex-direction:column;align-items:center">{dot(5)}</div>', unsafe_allow_html=True)
            with col_body:
                if step >= 5:
                    st.markdown(f'<div class="nb-step-title">Model export & generate</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:.8rem;color:{TEXT2};margin:.4rem 0 .75rem">Final section mein kya include karna hai:</div>', unsafe_allow_html=True)

                    ex = st.session_state.nb_export
                    ex["save_model"]    = st.checkbox("💾 Save model (.pkl)",           value=ex["save_model"],    key="ex_save")
                    ex["load_predict"]  = st.checkbox("🔮 Load model & predict example", value=ex["load_predict"],  key="ex_pred")
                    ex["summary_table"] = st.checkbox("📋 Session summary table",        value=ex["summary_table"], key="ex_sum")
                    st.session_state.nb_export = ex

                    # ── Summary of selections ──
                    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:.8rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:{TEXT3};margin-bottom:.75rem">📋 Notebook Summary</div>', unsafe_allow_html=True)

                    eda   = st.session_state.nb_eda
                    pre   = st.session_state.nb_pre
                    train = st.session_state.nb_train
                    expt  = st.session_state.nb_export

                    eda_labels   = {"distributions":"Distributions","correlation":"Heatmap","missing":"Missing values","target_dist":"Target dist","boxplots":"Box plots","scatter_matrix":"Scatter matrix","cat_bars":"Cat bars","outlier_plot":"Outlier plot"}
                    pre_labels   = {"drop_dups":"Drop dups","handle_missing":"Handle missing","normalize":"Normalize","remove_outliers":"Remove outliers"}
                    train_labels = {"model_table":"Model table","bar_chart":"Bar chart","radar_chart":"Radar chart","feature_importance":"Feature importance"}
                    exp_labels   = {"save_model":"Save model","load_predict":"Load & predict","summary_table":"Summary table"}

                    def chips_html(d, labels, color):
                        out = ""
                        for k, v in d.items():
                            label = labels.get(k, k)
                            if v:
                                out += f'<span style="display:inline-flex;align-items:center;padding:3px 10px;border-radius:6px;font-size:.72rem;font-weight:700;margin:2px;background:{"rgba(74,222,128,0.12)" if color=="green" else "rgba(96,165,250,0.12)"};color:{"#4ade80" if color=="green" else "#60a5fa"};border:1px solid {"rgba(74,222,128,0.35)" if color=="green" else "rgba(96,165,250,0.35)"}">✓ {label}</span>'
                            else:
                                out += f'<span style="display:inline-flex;align-items:center;padding:3px 10px;border-radius:6px;font-size:.72rem;margin:2px;background:{BG3};color:{TEXT3};border:1px solid {BORDER}">✗ {label}</span>'
                        return out

                    sum_sections = [
                        (f"🎯 Problem type", f'<span style="color:{ACCENT1};font-weight:700">{st.session_state.nb_ptype.title()}</span>'),
                        ("🧬 EDA charts",    chips_html(eda,   eda_labels,   "green")),
                        ("🧹 Preprocessing", chips_html(pre,   pre_labels,   "blue")),
                        ("🏆 Result charts", chips_html(train, train_labels, "green")),
                        ("💾 Export",        chips_html(expt,  exp_labels,   "blue")),
                    ]

                    for section_title, content in sum_sections:
                        st.markdown(f"""
                        <div style="margin-bottom:.75rem">
                          <div style="font-size:.72rem;font-weight:800;color:{TEXT3};text-transform:uppercase;letter-spacing:.06em;margin-bottom:.3rem">{section_title}</div>
                          <div>{content}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

                    # ── Back + Generate buttons ──
                    nb5c1, nb5c2 = st.columns([1, 4])
                    with nb5c1:
                        if st.button("← Back", key="nb_back5"):
                            st.session_state.nb_step = 4; st.rerun()

                    with nb5c2:
                        # Check if training has been done
                        if st.session_state.results is not None:
                            try:
                                res_nb = st.session_state.results
                                mc_nb  = "Model" if "Model" in res_nb.columns else res_nb.columns[0]
                                nr_nb  = res_nb.select_dtypes(include=[np.number]).columns.tolist()
                                bn_nb  = res_nb.iloc[0][mc_nb]
                                mn_nb  = nr_nb[0] if nr_nb else "Score"
                                ts_nb  = res_nb.iloc[0][mn_nb] if nr_nb else 0

                                nb_bytes = generate_notebook(
                                    df              = df,
                                    target_col      = st.session_state.get("nb_target", df.columns[0]),
                                    problem_type    = st.session_state.nb_ptype,
                                    results_df      = res_nb,
                                    best_model_name = bn_nb,
                                    top_score       = ts_nb,
                                    metric_name     = mn_nb,
                                    dataset_name    = str(st.session_state.dataset_name or "dataset"),
                                    training_time   = st.session_state.training_time or 0,
                                    fold            = st.session_state.cv_fold or 5,
                                    normalize       = pre.get("normalize", True),
                                    train_size      = 0.8,
                                    eda_selections  = st.session_state.nb_eda,
                                    pre_selections  = st.session_state.nb_pre,
                                    train_selections= st.session_state.nb_train,
                                    export_selections=st.session_state.nb_export,
                                )
                                safe_nm = str(st.session_state.dataset_name or "dataset").replace(" ","_").replace(".","_")
                                st.download_button(
                                    label     = "📓 Generate & Download Notebook (.ipynb)",
                                    data      = nb_bytes,
                                    file_name = f"dataforge_{safe_nm}_{bn_nb.replace(' ','_')}.ipynb",
                                    mime      = "application/x-ipynb+json",
                                    key       = "nb_builder_download"
                                )
                                st.success("✅ Notebook ready! Click button to download.")
                            except Exception as nb_err:
                                st.error(f"❌ Error: {nb_err}")
                        else:
                            st.warning("⚠️ Pehle **Train Model** tab mein training complete karo — phir notebook generate hogi!")
                            st.info("💡 Training ke baad yahan wapas aao aur **Generate** button click karo.")

                else:
                    st.markdown(f'<div class="nb-step-title-idle">Model export & generate</div>', unsafe_allow_html=True)

else:
    # ── WELCOME SCREEN ──
    st.markdown(f"""
    <div style="text-align:center;min-height:40vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem 2rem" class="slide-up">
      <div style="font-size:3.8rem;font-weight:900;letter-spacing:-.04em;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 1rem;line-height:1.05">Drop Your Data.<br>We Do the Rest.</div>
      <p style="font-size:1.1rem;color:{TEXT2};max-width:560px;line-height:1.7;margin:0 0 2rem">DataForge ML Studio — zero-code AutoML. Upload a CSV, pick a target, hit train. Get a production model in minutes. <b style="color:{ACCENT1}">Everything is 100% free.</b></p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    feats = [
        ("🧬","Smart EDA","Correlation heatmaps, distribution explorer, scatter builder, and missing value charts."),
        ("⚡","AutoML Engine","15+ algorithms compared with k-fold cross-validation. Best model wins — automatically."),
        ("🎯","Smart Detection","Auto-detects regression vs classification. Warns about ID columns. Quick data cleaning."),
        ("🏆","Rich Results","Trophy banner, radar + scatter + bar charts, metric breakdown, model export (.pkl)."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], feats):
        with col:
            st.markdown(f"""
            <div class="feature-card slide-up">
              <div class="fc-icon">{icon}</div>
              <h3>{title}</h3><p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{"rgba(74,222,128,0.06)" if T=="dark" else "rgba(124,58,237,0.06)"};border:2px solid {"rgba(74,222,128,0.25)" if T=="dark" else "rgba(124,58,237,0.25)"};border-radius:20px;padding:2rem;text-align:center;margin-bottom:2rem">
      <div style="font-size:1.4rem;font-weight:900;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:1rem">🎁 Everything Free. No Login. No Limits.</div>
      <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:.75rem">
        {"".join(f'<span class="insight-chip" style="border-color:{ACCENT1};color:{ACCENT1}">{f}</span>' for f in ["✅ XGBoost","✅ LightGBM","✅ CatBoost","✅ 10-fold CV","✅ Model Export (.pkl)","✅ Notebook Export (.ipynb)","✅ 13+ Algorithms","✅ No Sign-up Required"])}
      </div>
    </div>
    <div style="text-align:center;color:{TEXT3};font-size:.82rem;padding-bottom:1.5rem">
      👈 Upload a CSV/Excel or load a sample dataset from the sidebar to get started
    </div>
    """, unsafe_allow_html=True)
