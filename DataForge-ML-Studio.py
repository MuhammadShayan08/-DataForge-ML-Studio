# =====================================================================
#  DataForge ML Studio — All Features Free, No Login, No Admin
#  Notebook Builder Tab: Pro UI with 2-column stepper layout
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
import warnings, time, io, os, gc, json
from datetime import datetime
warnings.filterwarnings("ignore")

st.set_page_config(page_title="DataForge ML Studio", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

MAX_ROWS_TRAINING   = 5_000
MAX_ROWS_WARNING    = 2_000
SAMPLE_RANDOM_STATE = 42

ALL_CLF_MODELS = ["lr","dt","rf","et","ridge","knn","nb","ada","xgboost","lightgbm","catboost","gbc","lda"]
ALL_REG_MODELS = ["lr","dt","rf","et","ridge","lasso","knn","ada","en","xgboost","lightgbm","catboost","gbr","br"]

# ─────────────────────────────────────────────
#  STYLED MARKDOWN GENERATOR
# ─────────────────────────────────────────────
def make_styled_markdown_cells(
    df, target_col, problem_type, dataset_name,
    best_model_name, top_score, metric_name,
    author_name="", author_title="", author_quote="",
    author_email="", author_linkedin="", author_github="",
    author_kaggle="", author_facebook=""
):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    null_pct = round(df.isnull().sum().sum() / df.size * 100, 2)
    dup_cnt  = int(df.duplicated().sum())
    rows, cols = len(df), len(df.columns)
    task_label = "Classification" if problem_type == "classification" else "Regression"
    task_emoji = "🎯" if problem_type == "classification" else "📈"

    title_cell = f"""<p style="background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
         font-family: 'Montserrat', sans-serif; font-size: 26px; text-align: center;
         color: #ffffff; padding: 24px 42px; border-radius: 30px; border: 3px solid #a78bfa;
         box-shadow: 0 10px 25px rgba(102,126,234,0.3); letter-spacing: 1.2px;
         word-spacing: 4px; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); margin: 10px 0 20px;">
  ⚡ DataForge ML Studio — AutoML Notebook<br>
  <span style="font-size:16px;font-weight:400;opacity:0.9;">Dataset: {dataset_name} &nbsp;|&nbsp; Target: {target_col} &nbsp;|&nbsp; Task: {task_label}</span>
</p>
<div style="display:flex; gap:12px; flex-wrap:wrap; font-family:'Montserrat',sans-serif; margin: 0 2px 18px; justify-content: center;">
  <span style="background:#fff0f5; border:1px solid #ffc9e0; color:#8b1a5e; padding:8px 16px; border-radius:999px; font-size:13px; font-weight:600;">{task_emoji} <b>Task:</b> {task_label}</span>
  <span style="background:#f0f8ff; border:1px solid #b8d8ff; color:#1a4d8b; padding:8px 16px; border-radius:999px; font-size:13px; font-weight:600;">📊 <b>Dataset:</b> {dataset_name}</span>
  <span style="background:#fff9f0; border:1px solid #ffd9a8; color:#8b5a00; padding:8px 16px; border-radius:999px; font-size:13px; font-weight:600;">🎯 <b>Target:</b> {target_col}</span>
  <span style="background:#f0fff4; border:1px solid #9ae6b4; color:#22543d; padding:8px 16px; border-radius:999px; font-size:13px; font-weight:600;">🏆 <b>Best Model:</b> {best_model_name}</span>
  <span style="background:#f5f5f5; border:1px solid #d4d4d4; color:#404040; padding:8px 16px; border-radius:999px; font-size:13px; font-weight:600;">📅 <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d')}</span>
</div>"""

    overview_cell = f"""<div style="font-family:'Montserrat',sans-serif; background:linear-gradient(135deg, #e0f7ff 0%, #f0f9ff 100%); padding:20px 22px; border-radius:16px; border-left: 6px solid #667eea; box-shadow: 0 5px 18px rgba(102,126,234,0.15); color:#2c2c2c; margin: 0 0 20px;">
  🗂️ <b style="font-size:17px;">Dataset Overview:</b>
  <ul style="margin:12px 0 0 20px; line-height:2; font-size:15px;">
    <li>📋 <b>Total Rows:</b> {rows:,} &nbsp;|&nbsp; <b>Columns:</b> {cols}</li>
    <li>🔢 <b>Numerical Features:</b> {len(num_cols)} &nbsp;|&nbsp; 🔤 <b>Categorical Features:</b> {len(cat_cols)}</li>
    <li>❓ <b>Missing Values:</b> {null_pct}% &nbsp;|&nbsp; 🗑️ <b>Duplicate Rows:</b> {dup_cnt}</li>
    <li>🎯 <b>Target Column:</b> <code>{target_col}</code> &nbsp;|&nbsp; <b>Problem Type:</b> {task_label}</li>
    <li>🏆 <b>Best Model:</b> {best_model_name} &nbsp;|&nbsp; <b>{metric_name}:</b> {top_score:.4f}</li>
  </ul>
</div>"""

    findings_cell = f"""<div style="font-family:'Montserrat',sans-serif; background:linear-gradient(135deg, #e3f2fd 0%, #f7fbff 100%); padding:20px 22px; border-radius:16px; border:2px solid #90caf9; box-shadow:0 6px 20px rgba(33,150,243,0.12); color:#2a2a2a; margin-bottom:20px;">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;"><span style="font-size:22px;">🔥</span><h3 style="margin:0; font-size:18px; color:#1565c0; font-weight:700;">Key Findings</h3></div>
  <ul style="margin:10px 0 0 20px; padding:0; line-height:1.9; font-size:15px;">
    <li>📊 <b>Dataset Shape:</b> {rows:,} rows × {cols} columns</li>
    <li>🔢 <b>Feature Mix:</b> {len(num_cols)} numerical + {len(cat_cols)} categorical features</li>
    <li>{'✅' if null_pct == 0 else '⚠️'} <b>Data Quality:</b> {'Clean — no missing values' if null_pct == 0 else f'{null_pct}% missing values detected'}</li>
    <li>🏆 <b>Best Algorithm:</b> {best_model_name} outperformed all other models</li>
    <li>📈 <b>Performance Score ({metric_name}):</b> {top_score:.4f}</li>
  </ul>
</div>"""

    learn_cell = f"""<div style="font-family:'Montserrat',sans-serif; background:linear-gradient(135deg, #f3f7ff 0%, #f9fbff 100%); padding:20px 22px; border-radius:16px; border:2px solid #c7d7ff; box-shadow:0 6px 20px rgba(63,81,181,0.12); color:#2a2a2a; margin-bottom:20px;">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;"><span style="font-size:22px;">📚</span><h3 style="margin:0; font-size:18px; color:#283593; font-weight:700;">What You'll Learn</h3></div>
  <ul style="margin:10px 0 0 20px; padding:0; line-height:1.9; font-size:15px;">
    <li>🔍 <b>EDA Techniques:</b> Explore and visualize a {'classification' if problem_type=='classification' else 'regression'} dataset</li>
    <li>🧹 <b>Data Preprocessing:</b> Handle missing values, duplicates, and normalization</li>
    <li>🤖 <b>AutoML with PyCaret:</b> Compare {len(ALL_CLF_MODELS if problem_type=='classification' else ALL_REG_MODELS)}+ algorithms in one line</li>
    <li>🏆 <b>Model Selection:</b> Cross-validation picks the best model ({best_model_name})</li>
    <li>📊 <b>Result Visualization:</b> Bar charts, radar plots, and feature importance</li>
  </ul>
</div>"""

    pipeline_cell = f"""<div style="font-family:'Montserrat',sans-serif; background:#ffffff; padding:20px 22px; border-radius:16px; border:2px solid #e8e8e8; box-shadow:0 8px 24px rgba(0,0,0,0.08); color:#2a2a2a; margin-bottom:20px;">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;"><span style="font-size:22px;">📋</span><h3 style="margin:0; font-size:18px; color:#2c2c2c; font-weight:700;">AutoML Pipeline — Step by Step</h3></div>
  <ul style="margin:10px 0 0 20px; padding:0; line-height:2; font-size:15px;">
    <li>📦 <b>Step 1 — Data Loading:</b> Import <code>{dataset_name}</code> and inspect shape & types</li>
    <li>🔍 <b>Step 2 — EDA:</b> Visualize distributions, correlations, and target variable</li>
    <li>🧹 <b>Step 3 — Preprocessing:</b> Handle nulls, remove duplicates, normalize features</li>
    <li>⚙️ <b>Step 4 — PyCaret Setup:</b> Configure {task_label} environment with CV</li>
    <li>🤖 <b>Step 5 — Model Comparison:</b> Benchmark all algorithms automatically</li>
    <li>🏆 <b>Step 6 — Best Model:</b> {best_model_name} selected with {metric_name} = {top_score:.4f}</li>
    <li>💾 <b>Step 7 — Export:</b> Save model as .pkl for deployment</li>
  </ul>
</div>"""

    tech_cell = f"""<div style="font-family:'Montserrat',sans-serif; background:linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); padding:20px 22px; border-radius:16px; border:2px solid #81c784; box-shadow:0 6px 20px rgba(76,175,80,0.12); color:#2a2a2a; margin-bottom:20px;">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;"><span style="font-size:22px;">🛠️</span><h3 style="margin:0; font-size:18px; color:#2e7d32; font-weight:700;">Tech Stack Used</h3></div>
  <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:12px;">
    <span style="background:#fff; border:2px solid #607d8b; color:#455a64; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;">🐍 Python</span>
    <span style="background:#fff; border:2px solid #3f51b5; color:#283593; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;">🤖 PyCaret</span>
    <span style="background:#fff; border:2px solid #ff9800; color:#e65100; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;">🐼 Pandas</span>
    <span style="background:#fff; border:2px solid #4caf50; color:#2e7d32; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;">🔢 NumPy</span>
    <span style="background:#fff; border:2px solid #9c27b0; color:#6a1b9a; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;">📊 Plotly</span>
    <span style="background:#fff; border:2px solid #f44336; color:#c62828; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;">⚡ XGBoost</span>
    <span style="background:#fff; border:2px solid #00bcd4; color:#006064; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;">💡 LightGBM</span>
    <span style="background:#fff; border:2px solid #ff5722; color:#bf360c; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;">🐱 CatBoost</span>
  </div>
</div>"""

    def section_banner(title, color1="#667eea", color2="#764ba2"):
        return f"""<p style="background: linear-gradient(90deg, {color1}, {color2}); font-family: 'Montserrat', sans-serif; font-size: 22px; text-align: center; color: #ffffff; padding: 18px 42px; border-radius: 30px; border: 3px solid {color1}99; box-shadow: 0 8px 20px {color1}44; letter-spacing: 1.1px; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); margin: 30px 0 18px;">{title}</p>"""

    connect_links = ""
    if author_email:    connect_links += f'<li><a href="mailto:{author_email}" style="background:linear-gradient(135deg,#fff3f3,#ffe8e8);padding:11px 18px;border-radius:12px;text-decoration:none;color:#d32f2f;font-weight:600;font-size:14px;border:2px solid #ffcdd2;display:inline-block;">📧 Email</a></li>'
    if author_linkedin: connect_links += f'<li><a href="{author_linkedin}" style="background:linear-gradient(135deg,#e3f2fd,#e8eaf6);padding:11px 18px;border-radius:12px;text-decoration:none;color:#1565c0;font-weight:600;font-size:14px;border:2px solid #bbdefb;display:inline-block;">🔗 LinkedIn</a></li>'
    if author_github:   connect_links += f'<li><a href="{author_github}" style="background:linear-gradient(135deg,#f5f5f5,#eeeeee);padding:11px 18px;border-radius:12px;text-decoration:none;color:#212121;font-weight:600;font-size:14px;border:2px solid #e0e0e0;display:inline-block;">💻 GitHub</a></li>'
    if author_kaggle:   connect_links += f'<li><a href="{author_kaggle}" style="background:linear-gradient(135deg,#e0f7fa,#e1f5fe);padding:11px 18px;border-radius:12px;text-decoration:none;color:#00838f;font-weight:600;font-size:14px;border:2px solid #b2ebf2;display:inline-block;">🧠 Kaggle</a></li>'
    if author_facebook: connect_links += f'<li><a href="{author_facebook}" style="background:linear-gradient(135deg,#e7f0ff,#dde7ff);padding:11px 18px;border-radius:12px;text-decoration:none;color:#0d47a1;font-weight:600;font-size:14px;border:2px solid #c5cae9;display:inline-block;">📘 Facebook</a></li>'

    quote_html = f"""<blockquote style="background:linear-gradient(135deg,#e8f5e9,#f1f8e9);border-left:5px solid #4caf50;padding:16px 20px;font-style:italic;border-radius:12px;color:#2c2c2c;margin:16px 0 0;font-size:15px;line-height:1.7;">💡 "{author_quote}"</blockquote>""" if author_quote else ""

    author_cell = ""
    if author_name:
        author_cell = f"""{section_banner("🧑‍💻 About the Developer", "#667eea", "#764ba2")}
<div style="font-family:'Montserrat',sans-serif;background:linear-gradient(135deg,#ffffff,#f8f9fa);border-radius:18px;padding:28px;border:2px solid #e0e7ff;box-shadow:0 8px 24px rgba(102,126,234,0.12);color:#2c2c2c;margin-bottom:25px;">
  <p style="font-size:16px;line-height:1.8;margin:0 0 14px;">Hi, I'm <strong style="color:#667eea;font-size:17px;">{author_name}</strong>{f' — <strong>{author_title}</strong>' if author_title else ''}.</p>
  <p style="font-size:16px;line-height:1.8;margin:0 0 16px;">This notebook was generated using <strong>DataForge ML Studio</strong> — an AutoML platform that makes machine learning accessible to everyone.</p>
  {quote_html}
</div>"""
        if connect_links:
            author_cell += f"""{section_banner("🌐 Connect With Me", "#667eea", "#f093fb")}
<div style="font-family:'Montserrat',sans-serif;background:linear-gradient(135deg,#ffffff,#fafcff);border-radius:18px;padding:28px;border:2px solid #e0e7ff;">
  <ul style="list-style:none;padding:0;margin:0;display:flex;flex-wrap:wrap;gap:12px;justify-content:center;">{connect_links}</ul>
</div>
<p style="text-align:center;font-family:'Montserrat',sans-serif;color:#666;font-size:13px;margin-top:30px;font-style:italic;">⭐ If you find this notebook valuable, please upvote and share feedback!</p>"""

    banners = {
        "imports":    section_banner("📦 Section 1 — Install & Import Libraries", "#667eea", "#764ba2"),
        "load":       section_banner("📂 Section 2 — Load Dataset", "#f093fb", "#f5576c"),
        "overview":   section_banner("🔍 Section 3 — Dataset Overview", "#4facfe", "#00f2fe"),
        "eda":        section_banner("🧬 Section 4 — Exploratory Data Analysis", "#43e97b", "#38f9d7"),
        "preprocess": section_banner("🧹 Section 5 — Data Preprocessing", "#fa709a", "#fee140"),
        "training":   section_banner("⚙️ Section 6 — AutoML Training with PyCaret", "#a18cd1", "#fbc2eb"),
        "results":    section_banner("🏆 Section 7 — Results & Visualizations", "#ffecd2", "#fcb69f"),
        "export":     section_banner("💾 Section 8 — Save & Export Model", "#667eea", "#43e97b"),
        "summary":    section_banner("📋 Section 9 — Session Summary", "#764ba2", "#667eea"),
    }
    return {"title":title_cell,"overview":overview_cell,"findings":findings_cell,"learn":learn_cell,"pipeline":pipeline_cell,"tech":tech_cell,"banners":banners,"author":author_cell}


# ─────────────────────────────────────────────
#  NOTEBOOK GENERATOR
# ─────────────────────────────────────────────
import json as _json

def generate_notebook(df, target_col, problem_type, results_df, best_model_name,
                      top_score, metric_name, dataset_name, training_time, fold,
                      normalize, train_size,
                      eda_selections=None, pre_selections=None,
                      train_selections=None, export_selections=None,
                      author_name="", author_title="", author_quote="",
                      author_email="", author_linkedin="", author_github="",
                      author_kaggle="", author_facebook=""):

    eda  = eda_selections or {"distributions":True,"correlation":True,"missing":True,"target_dist":True,"boxplots":True,"scatter_matrix":True,"cat_bars":True,"outlier_plot":True,"violin_plots":True,"feature_importance_eda":True,"skewness_kurtosis":True,"value_counts_table":True,"numeric_summary_styled":True,"categorical_summary":True,"outlier_table":True,"iqr_analysis":True}
    pre  = pre_selections or {"drop_dups":True,"handle_missing":True,"normalize":True,"remove_outliers":False}
    trn  = train_selections or {"model_table":True,"bar_chart":True,"radar_chart":True,"feature_importance":True}
    exp  = export_selections or {"save_model":True,"load_predict":True,"summary_table":True}

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    null_pct = round(df.isnull().sum().sum() / df.size * 100, 2)
    dup_cnt  = int(df.duplicated().sum())

    styled = make_styled_markdown_cells(df=df,target_col=target_col,problem_type=problem_type,dataset_name=dataset_name,best_model_name=best_model_name,top_score=top_score,metric_name=metric_name,author_name=author_name,author_title=author_title,author_quote=author_quote,author_email=author_email,author_linkedin=author_linkedin,author_github=author_github,author_kaggle=author_kaggle,author_facebook=author_facebook)
    banners = styled["banners"]

    def code_cell(source, cell_id=""):
        return {"cell_type":"code","execution_count":None,"id":cell_id or source[:8].replace(" ","_"),"metadata":{},"outputs":[],"source":source.strip()}
    def md_cell(source, cell_id=""):
        return {"cell_type":"markdown","id":cell_id or source[:8].replace(" ","_"),"metadata":{},"source":source.strip()}

    sp = "clf" if problem_type=="classification" else "reg"
    pm = "pycaret.classification" if problem_type=="classification" else "pycaret.regression"
    ncr = repr(num_cols); ccr = repr(cat_cols)
    try: rs = results_df.to_string(index=False)
    except: rs = str(results_df)

    cells = []
    cells.append(md_cell(styled["title"],"title"))
    cells.append(md_cell(styled["overview"],"overview"))
    cells.append(md_cell(styled["findings"],"findings"))
    cells.append(md_cell(styled["learn"],"learn"))
    cells.append(md_cell(styled["pipeline"],"pipeline"))
    cells.append(md_cell(styled["tech"],"tech"))

    cells.append(md_cell(banners["imports"],"s1"))
    cells.append(code_cell(f"""# !pip install pycaret plotly pandas numpy openpyxl scipy
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from {pm} import (setup as {sp}_setup, compare_models as {sp}_compare, pull as {sp}_pull, save_model as {sp}_save, load_model as {sp}_load)
print("✅ Libraries imported successfully!")""","s1_imp"))

    cells.append(md_cell(banners["load"],"s2"))
    cells.append(code_cell(f"""# df = pd.read_csv("your_file.csv")
TARGET = "{target_col}"
print(f"✅ Dataset loaded | Shape: {{df.shape}} | Target: {{TARGET}}")
df.head()""","s2_load"))

    cells.append(md_cell(banners["overview"],"s3"))
    cells.append(code_cell(f"""print(f"📊 Rows: {len(df):,} | Cols: {len(df.columns)} | Missing: {null_pct}% | Duplicates: {dup_cnt}")
df.info()""","s3_info"))
    cells.append(code_cell("df.describe(include='all').round(3)","s3_desc"))

    eda_added = False
    def eda_banner():
        nonlocal eda_added
        if not eda_added:
            cells.append(md_cell(banners["eda"],"s4"))
            eda_added = True

    if eda.get("missing"):
        eda_banner()
        cells.append(code_cell("""missing = df.isnull().sum(); missing = missing[missing>0].sort_values(ascending=False)
if len(missing)>0:
    fig = px.bar(x=missing.index,y=missing.values,title="❓ Missing Values per Column",color=missing.values,color_continuous_scale=["#4ade80","#fbbf24","#f87171"])
    fig.update_layout(template="plotly_dark",height=350); fig.show()
else: print("✅ No missing values found!")""","s4_miss"))

    if eda.get("distributions") and num_cols:
        eda_banner()
        cells.append(code_cell(f"""num_cols={ncr}; cols_to_plot=num_cols[:9]; n_c=min(3,len(cols_to_plot)); n_r=(len(cols_to_plot)+n_c-1)//n_c
fig=make_subplots(rows=n_r,cols=n_c,subplot_titles=cols_to_plot)
for i,col in enumerate(cols_to_plot):
    fig.add_trace(go.Histogram(x=df[col],name=col,marker_color="#4ade80",opacity=0.8),row=i//n_c+1,col=i%n_c+1)
fig.update_layout(template="plotly_dark",height=280*n_r,title_text="📊 Feature Distributions",showlegend=False); fig.show()""","s4_hist"))

    if eda.get("violin_plots") and num_cols:
        eda_banner()
        cells.append(code_cell(f"""num_cols={ncr}; cols_vln=num_cols[:8]; n_c=min(2,len(cols_vln)); n_r=(len(cols_vln)+n_c-1)//n_c
fig=make_subplots(rows=n_r,cols=n_c,subplot_titles=cols_vln)
colors=["#4ade80","#60a5fa","#c084fc","#fbbf24","#f87171","#fb923c","#34d399","#a78bfa"]
for i,col in enumerate(cols_vln):
    fig.add_trace(go.Violin(y=df[col].dropna(),name=col,box_visible=True,meanline_visible=True,points="outliers",fillcolor=colors[i%len(colors)],line_color=colors[i%len(colors)],opacity=0.75),row=i//n_c+1,col=i%n_c+1)
fig.update_layout(template="plotly_dark",height=320*n_r,title_text="🎻 Violin Plots",showlegend=False); fig.show()""","s4_violin"))

    if eda.get("boxplots") and num_cols:
        eda_banner()
        cells.append(code_cell(f"""num_cols={ncr}; cols_bp=num_cols[:6]; n_c=min(3,len(cols_bp)); n_r=(len(cols_bp)+n_c-1)//n_c
fig=make_subplots(rows=n_r,cols=n_c,subplot_titles=cols_bp)
for i,col in enumerate(cols_bp):
    fig.add_trace(go.Box(y=df[col],name=col,marker_color="#60a5fa"),row=i//n_c+1,col=i%n_c+1)
fig.update_layout(template="plotly_dark",height=280*n_r,title_text="📦 Box Plots",showlegend=False); fig.show()""","s4_box"))

    if eda.get("correlation") and len(num_cols)>=2:
        eda_banner()
        cells.append(code_cell(f"""corr=df[{ncr}[:15]].corr().round(2)
fig=go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,colorscale=[[0,"#f87171"],[0.5,"#1c1c1c"],[1,"#4ade80"]],zmid=0,text=corr.values.round(2),texttemplate="%{{text}}",textfont=dict(size=9)))
fig.update_layout(template="plotly_dark",height=520,title="🔥 Correlation Heatmap"); fig.show()""","s4_corr"))

    if eda.get("skewness_kurtosis") and num_cols:
        eda_banner()
        cells.append(code_cell(f"""from scipy.stats import skew,kurtosis
num_cols={ncr}; sk_data=[]
for col in num_cols:
    s=df[col].dropna()
    sk_data.append({{"Feature":col,"Skewness":round(skew(s),3),"Kurtosis":round(kurtosis(s),3),"Mean":round(s.mean(),3),"Std":round(s.std(),3)}})
sk_df=pd.DataFrame(sk_data).sort_values("Skewness",key=abs,ascending=False)
print("📐 Skewness & Kurtosis:"); print(sk_df.to_string(index=False))
fig=make_subplots(rows=1,cols=2,subplot_titles=["Skewness","Kurtosis"])
colors_sk=["#f87171" if abs(v)>1 else "#fbbf24" if abs(v)>0.5 else "#4ade80" for v in sk_df["Skewness"]]
fig.add_trace(go.Bar(x=sk_df["Skewness"],y=sk_df["Feature"],orientation="h",marker_color=colors_sk,name="Skewness"),row=1,col=1)
fig.add_trace(go.Bar(x=sk_df["Kurtosis"],y=sk_df["Feature"],orientation="h",marker_color="#c084fc",name="Kurtosis"),row=1,col=2)
fig.update_layout(template="plotly_dark",height=max(300,len(sk_df)*30),title_text="📐 Skewness & Kurtosis",showlegend=False); fig.show()""","s4_skew"))

    if eda.get("numeric_summary_styled") and num_cols:
        eda_banner()
        cells.append(code_cell(f"""num_cols={ncr}; summary_rows=[]
for col in num_cols:
    s=df[col].dropna(); Q1,Q3=s.quantile(0.25),s.quantile(0.75); IQR=Q3-Q1
    outliers=((s<Q1-1.5*IQR)|(s>Q3+1.5*IQR)).sum()
    summary_rows.append({{"Feature":col,"Count":int(s.count()),"Missing%":round(df[col].isnull().mean()*100,1),"Mean":round(s.mean(),4),"Std":round(s.std(),4),"Min":round(s.min(),4),"Median":round(s.median(),4),"Max":round(s.max(),4),"IQR":round(IQR,4),"Outliers":int(outliers),"Outlier%":round(outliers/len(df)*100,1),"Skewness":round(s.skew(),3)}})
summary_df=pd.DataFrame(summary_rows)
summary_df.style.background_gradient(subset=["Missing%","Outlier%"],cmap="RdYlGn_r").background_gradient(subset=["Skewness"],cmap="RdBu").format(precision=3)""","s4_num_summary"))

    if eda.get("scatter_matrix") and len(num_cols)>=2:
        eda_banner()
        cells.append(code_cell(f"""scatter_cols={ncr}[:5]
fig=px.scatter_matrix(df[scatter_cols],title="🔵 Scatter Matrix",color_discrete_sequence=["#4ade80"])
fig.update_layout(template="plotly_dark",height=600); fig.show()""","s4_scat"))

    if eda.get("target_dist"):
        eda_banner()
        cells.append(code_cell(f"""target_series=df["{target_col}"]
if "{problem_type}"=="classification":
    vc=target_series.value_counts()
    fig=px.pie(values=vc.values,names=vc.index.astype(str),title="🎯 Target Distribution — {target_col}",color_discrete_sequence=["#4ade80","#60a5fa","#c084fc","#fbbf24","#f87171"])
else:
    fig=px.histogram(df,x="{target_col}",nbins=40,title="🎯 Target Distribution — {target_col}",color_discrete_sequence=["#4ade80"])
fig.update_layout(template="plotly_dark",height=380); fig.show()""","s4_tgt"))

    if eda.get("cat_bars") and cat_cols:
        eda_banner()
        cells.append(code_cell(f"""cat_cols={ccr}
for col in cat_cols[:4]:
    vc=df[col].value_counts().head(12)
    fig=px.bar(x=vc.index.astype(str),y=vc.values,title=f"📋 Value Counts — {{col}}",color=vc.values,color_continuous_scale=["#60a5fa","#4ade80"])
    fig.update_layout(template="plotly_dark",height=320,showlegend=False,coloraxis_showscale=False); fig.show()""","s4_cat"))

    if eda.get("categorical_summary") and cat_cols:
        eda_banner()
        cells.append(code_cell(f"""cat_cols={ccr}; cat_summary=[]
for col in cat_cols:
    s=df[col]; vc=s.value_counts()
    cat_summary.append({{"Feature":col,"Unique":s.nunique(),"Missing%":round(s.isnull().mean()*100,1),"Top Value":str(vc.index[0]) if len(vc)>0 else "N/A","Top Count":int(vc.iloc[0]) if len(vc)>0 else 0,"Top%":round(vc.iloc[0]/len(df)*100,1) if len(vc)>0 else 0,"Cardinality":"Low" if s.nunique()<=5 else "Medium" if s.nunique()<=20 else "High"}})
cat_df=pd.DataFrame(cat_summary); print("🔤 Categorical Summary:"); print(cat_df.to_string(index=False))
fig=px.bar(cat_df.sort_values("Unique",ascending=True),x="Unique",y="Feature",orientation="h",color="Unique",color_continuous_scale=["#4ade80","#fbbf24","#f87171"],title="🔤 Categorical Cardinality",text="Unique")
fig.update_layout(template="plotly_dark",height=max(280,len(cat_cols)*35),coloraxis_showscale=False); fig.show()""","s4_cat_summary"))

    if eda.get("outlier_plot") and num_cols:
        eda_banner()
        cells.append(code_cell(f"""num_cols={ncr}; outlier_info=[]
for col in num_cols:
    Q1,Q3=df[col].quantile(0.25),df[col].quantile(0.75); IQR=Q3-Q1
    n_out=((df[col]<Q1-1.5*IQR)|(df[col]>Q3+1.5*IQR)).sum()
    outlier_info.append({{"column":col,"outliers":int(n_out),"pct":round(n_out/len(df)*100,2)}})
out_df=pd.DataFrame(outlier_info).sort_values("outliers",ascending=False)
fig=px.bar(out_df,x="column",y="pct",title="⚠️ Outlier % per Column (IQR)",color="pct",color_continuous_scale=["#4ade80","#fbbf24","#f87171"])
fig.update_layout(template="plotly_dark",height=380); fig.show(); print(out_df.to_string(index=False))""","s4_out"))

    if eda.get("iqr_analysis") and num_cols:
        eda_banner()
        cells.append(code_cell(f"""num_cols={ncr}; iqr_rows=[]
for col in num_cols:
    s=df[col].dropna(); Q1,Q3=s.quantile(0.25),s.quantile(0.75); IQR=Q3-Q1; lower=Q1-1.5*IQR; upper=Q3+1.5*IQR
    n_low=(s<lower).sum(); n_high=(s>upper).sum()
    iqr_rows.append({{"Feature":col,"Q1":round(Q1,3),"Q3":round(Q3,3),"IQR":round(IQR,3),"Lower":round(lower,3),"Upper":round(upper,3),"Low Out":int(n_low),"High Out":int(n_high),"Total":int(n_low+n_high),"Outlier%":round((n_low+n_high)/len(df)*100,2)}})
iqr_df=pd.DataFrame(iqr_rows).sort_values("Total",ascending=False)
print("📏 IQR Analysis:"); print(iqr_df.to_string(index=False))
fig=go.Figure()
fig.add_trace(go.Bar(x=iqr_df["Feature"],y=iqr_df["Low Out"],name="Low Outliers",marker_color="#60a5fa"))
fig.add_trace(go.Bar(x=iqr_df["Feature"],y=iqr_df["High Out"],name="High Outliers",marker_color="#f87171"))
fig.update_layout(barmode="stack",template="plotly_dark",height=380,title="📏 IQR Outlier Analysis"); fig.show()""","s4_iqr"))

    if eda.get("feature_importance_eda") and num_cols and len(num_cols)>=2:
        eda_banner()
        cells.append(code_cell(f"""from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
X_quick=df.select_dtypes(include="number").drop(columns=["{target_col}"],errors="ignore").fillna(0)
y_quick=df["{target_col}"].fillna(df["{target_col}"].mode()[0])
if len(X_quick.columns)>0 and len(X_quick)>10:
    try:
        if "{problem_type}"=="classification":
            le=LabelEncoder(); y_enc=le.fit_transform(y_quick.astype(str))
            rf=RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=-1,max_depth=5); rf.fit(X_quick,y_enc)
        else:
            rf=RandomForestRegressor(n_estimators=50,random_state=42,n_jobs=-1,max_depth=5); rf.fit(X_quick,y_quick)
        fi=pd.DataFrame({{"Feature":X_quick.columns,"Importance":rf.feature_importances_}}).sort_values("Importance",ascending=True)
        fig=px.bar(fi,x="Importance",y="Feature",orientation="h",title="⭐ Quick Feature Importance (RF-EDA)",color="Importance",color_continuous_scale=["#60a5fa","#4ade80","#fbbf24"],text=fi["Importance"].round(4))
        fig.update_layout(template="plotly_dark",height=max(300,len(fi)*30),coloraxis_showscale=False); fig.update_traces(textposition="outside"); fig.show()
    except Exception as e: print(f"Could not compute: {{e}}")""","s4_fi_eda"))

    if eda.get("outlier_table") and num_cols:
        eda_banner()
        cells.append(code_cell(f"""num_cols={ncr}; all_outlier_mask=pd.Series([False]*len(df))
for col in num_cols:
    Q1,Q3=df[col].quantile(0.25),df[col].quantile(0.75); IQR=Q3-Q1
    all_outlier_mask=all_outlier_mask|((df[col]<Q1-1.5*IQR)|(df[col]>Q3+1.5*IQR))
outlier_rows=df[all_outlier_mask]
print(f"🔎 Rows with outliers: {{len(outlier_rows):,}} / {{len(df):,}} ({{len(outlier_rows)/len(df)*100:.1f}}%)")
outlier_rows.head(10)""","s4_out_table"))

    if eda.get("value_counts_table") and cat_cols:
        eda_banner()
        cells.append(code_cell(f"""cat_cols={ccr}
for col in cat_cols:
    vc=df[col].value_counts(dropna=False)
    vc_df=pd.DataFrame({{"Value":vc.index.astype(str),"Count":vc.values,"Percentage":(vc.values/len(df)*100).round(2)}})
    print(f"\\n📋 Value Counts — {{col}} ({{df[col].nunique()}} unique):")
    print(vc_df.head(15).to_string(index=False))""","s4_vc_table"))

    # PREPROCESSING
    pre_steps = []
    if pre.get("drop_dups"):
        pre_steps.append("""before=len(df); df=df.drop_duplicates().reset_index(drop=True)
print(f"✅ Duplicates removed: {before-len(df)} | New shape: {df.shape}")""")
    if pre.get("handle_missing"):
        pre_steps.append("""nc=df.isnull().sum(); nc=nc[nc>0]
print("Missing columns:",nc.to_string() if len(nc) else "None ✅")
print("ℹ️ PyCaret will auto-impute during setup()")""")
    if pre.get("normalize"):
        pre_steps.append("""print("✅ Normalization will be applied in PyCaret setup(normalize=True)")""")
    if pre.get("remove_outliers") and problem_type=="regression":
        pre_steps.append("""from scipy import stats
num_cols_pre=df.select_dtypes(include='number').columns.drop('target',errors='ignore')
z_scores=stats.zscore(df[num_cols_pre].fillna(df[num_cols_pre].median()))
df=df[(abs(z_scores)<3).all(axis=1)].reset_index(drop=True)
print(f"✅ After outlier removal: {df.shape}")""")
    if pre_steps:
        cells.append(md_cell(banners["preprocess"],"s5"))
        for i,step_code in enumerate(pre_steps):
            cells.append(code_cell(step_code,f"s5_pre{i}"))

    # TRAINING
    cells.append(md_cell(banners["training"],"s6"))
    cells.append(code_cell(f"""{sp}_setup(data=df,target="{target_col}",train_size={train_size},fold={fold},normalize={normalize},verbose=True,session_id=42,n_jobs=-1)
print("✅ PyCaret setup complete!")""","s6_setup"))
    cells.append(code_cell(f"""best_model={sp}_compare(verbose=True,n_select=1)
results={sp}_pull()
print("\\n✅ Training complete! Best model:",best_model)""","s6_cmp"))

    # RESULTS
    res_added = False
    def res_banner():
        nonlocal res_added
        if not res_added:
            cells.append(md_cell(banners["results"],"s7"))
            res_added = True

    if trn.get("model_table"):
        res_banner()
        cells.append(code_cell(f"""print("DataForge ML Studio — Model Comparison:")
print(\"\"\"{rs}\"\"\")
results_df={sp}_pull(); results_df""","s7_res"))
    if trn.get("bar_chart"):
        res_banner()
        cells.append(code_cell(f"""results_df={sp}_pull(); mc=results_df.select_dtypes(include='number').columns[0]; ml=results_df.columns[0]; top6=results_df.head(6)
colors=["#4ade80"]+["#1c1c1c"]*5
fig=go.Figure(go.Bar(x=top6[mc],y=top6[ml],orientation="h",marker_color=colors,text=top6[mc].round(4),textposition="inside",textfont=dict(size=11,color="white")))
fig.update_layout(template="plotly_dark",height=380,title=f"🏆 Top Models — {{mc}}",yaxis=dict(autorange="reversed")); fig.show()""","s7_bar"))
    if trn.get("radar_chart"):
        res_banner()
        cells.append(code_cell(f"""results_df={sp}_pull(); nm=results_df.select_dtypes(include='number').columns[:6].tolist(); bv=results_df.iloc[0][nm]; norm=(bv-bv.min())/(bv.max()-bv.min()+1e-9)
fig=go.Figure(go.Scatterpolar(r=list(norm.values)+[norm.values[0]],theta=list(norm.index)+[norm.index[0]],fill="toself",fillcolor="rgba(74,222,128,0.18)",line=dict(color="#4ade80",width=2.5),marker=dict(size=6,color="#4ade80")))
fig.update_layout(template="plotly_dark",height=400,title="🕸️ Best Model Metrics — {best_model_name}",polar=dict(radialaxis=dict(visible=True,range=[0,1]))); fig.show()""","s7_radar"))
    if trn.get("feature_importance"):
        res_banner()
        cells.append(code_cell(f"""try:
    importances=best_model.feature_importances_
    feature_names=df.drop(columns=["{target_col}"]).columns.tolist()
    fi_df=pd.DataFrame({{"feature":feature_names,"importance":importances}}).sort_values("importance",ascending=True).tail(15)
    fig=px.bar(fi_df,x="importance",y="feature",orientation="h",title="⭐ Feature Importance — {best_model_name}",color="importance",color_continuous_scale=["#60a5fa","#4ade80"])
    fig.update_layout(template="plotly_dark",height=450); fig.show()
except Exception as e: print(f"Not available: {{e}}")""","s7_feat"))

    # EXPORT
    exp_added = False
    def exp_banner():
        nonlocal exp_added
        if not exp_added:
            cells.append(md_cell(banners["export"],"s8"))
            exp_added = True

    if exp.get("save_model"):
        exp_banner()
        cells.append(code_cell(f"""{sp}_save(best_model,"best_model_dataforge")
print("✅ Model saved: best_model_dataforge.pkl")""","s8_save"))
    if exp.get("load_predict"):
        exp_banner()
        cells.append(code_cell(f"""loaded_model={sp}_load("best_model_dataforge")
print("✅ Loaded:",type(loaded_model))
# from {pm} import predict_model
# preds=predict_model(loaded_model,data=df.drop(columns=[TARGET]).head(5)); print(preds)""","s8_load"))

    if exp.get("summary_table"):
        cells.append(md_cell(banners["summary"],"s9"))
        cells.append(md_cell(f"""| Property | Value |
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

> ⚡ Generated by **DataForge ML Studio** — AutoML for everyone, 100% free.""","s9_sum"))

    if styled["author"]:
        cells.append(md_cell(styled["author"],"s_author"))

    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10.0"}},"cells":cells}
    return _json.dumps(notebook,indent=2,ensure_ascii=False).encode("utf-8")


# ─────────────────────────────────────────────
#  MEMORY HELPERS
# ─────────────────────────────────────────────
def get_memory_usage_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss/1024/1024
    except: return 0.0

def force_gc():
    gc.collect(); gc.collect()
    try:
        import ctypes; ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except: pass

def smart_sample(df, target_col, max_rows=MAX_ROWS_TRAINING):
    if len(df)<=max_rows: return df
    try:
        target_series=df[target_col]
        if target_series.dtype=="object" or target_series.nunique()<=20:
            class_counts=target_series.value_counts(); valid_classes=class_counts[class_counts>=2].index
            df_filtered=df[target_series.isin(valid_classes)]
            if len(df_filtered)>=max_rows:
                from sklearn.model_selection import train_test_split
                _,sampled=train_test_split(df_filtered,test_size=max_rows/len(df_filtered),stratify=df_filtered[target_col],random_state=SAMPLE_RANDOM_STATE)
                return sampled.reset_index(drop=True)
    except: pass
    return df.sample(n=max_rows,random_state=SAMPLE_RANDOM_STATE).reset_index(drop=True)

def safe_fold_count(df,target_col,requested_fold,problem_type):
    if problem_type!="classification": return requested_fold
    try:
        min_class_count=df[target_col].value_counts().min()
        return max(2,min(requested_fold,int(min_class_count)))
    except: return max(2,requested_fold)

def drop_rare_classes(df,target_col,min_count=2):
    try:
        class_counts=df[target_col].value_counts(); rare=class_counts[class_counts<min_count].index.tolist()
        if rare:
            df_clean=df[~df[target_col].isin(rare)].reset_index(drop=True)
            warn=f"⚠️ **{len(rare)} rare class(es) removed** (<{min_count} samples): `{'`, `'.join([str(r) for r in rare[:5]])}`."
            return df_clean,warn
    except: pass
    return df,None

def run_memory_safe_training(df,target_col,problem_type,train_size,fold,normalize,remove_out,max_models=None):
    warnings_list=[]; t0=time.time()
    if problem_type=="classification":
        df,rare_warn=drop_rare_classes(df,target_col,min_count=2)
        if rare_warn: warnings_list.append(rare_warn)
        if len(df)<10: raise ValueError("Dataset mein bahut kam samples hain.")
    original_rows=len(df)
    if original_rows>MAX_ROWS_TRAINING:
        df_train=smart_sample(df,target_col,MAX_ROWS_TRAINING)
        warnings_list.append(f"⚠️ Dataset {original_rows:,} rows — auto-sampled to **{MAX_ROWS_TRAINING:,} rows**.")
    elif original_rows>MAX_ROWS_WARNING:
        df_train=df.copy()
        warnings_list.append(f"💡 Dataset has {original_rows:,} rows — training will proceed.")
    else: df_train=df.copy()
    safe_fold=safe_fold_count(df_train,target_col,fold,problem_type)
    if safe_fold!=fold: warnings_list.append(f"⚠️ **CV Folds reduced from {fold} to {safe_fold}**.")
    include_models=ALL_CLF_MODELS if problem_type=="classification" else ALL_REG_MODELS
    if max_models and max_models<len(include_models): include_models=include_models[:max_models]
    mem_before=get_memory_usage_mb()
    if mem_before>400: force_gc()
    setup_kwargs=dict(data=df_train,target=target_col,train_size=float(train_size),fold=int(safe_fold),normalize=normalize,verbose=False,html=False,session_id=42,n_jobs=1,use_gpu=False)
    if remove_out and problem_type=="regression" and len(df_train)>100: setup_kwargs["remove_outliers"]=True
    try:
        if problem_type=="classification":
            clf_setup(**setup_kwargs); pull_fn,save_fn,cmp_fn=clf_pull,clf_save,clf_compare
        else:
            reg_setup(**setup_kwargs); pull_fn,save_fn,cmp_fn=reg_pull,reg_save,reg_compare
    except Exception as e:
        err=str(e).lower()
        if "memory" in err or "killed" in err: raise MemoryError(f"Out of memory — reduce dataset to {MAX_ROWS_WARNING:,} rows.")
        if "least populated" in err or "stratif" in err: raise ValueError("Cross-validation error: Kuch target classes mein bohut kam samples hain.")
        raise
    force_gc()
    try:
        best=cmp_fn(verbose=False,n_select=1,include=include_models,errors="ignore"); results=pull_fn()
    except MemoryError:
        force_gc(); light=["lr","dt","ridge"]
        warnings_list.append("⚠️ Memory issue — retrying with 3 lightest models.")
        best=cmp_fn(verbose=False,n_select=1,include=light); results=pull_fn()
    try: save_fn(best,"best_model")
    except: pass
    force_gc()
    elapsed=time.time()-t0
    return best,results,elapsed,warnings_list,len(df_train)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for k in ["data","problem_type","best_model","results","training_time","dataset_name","cv_fold","target_col"]:
    if k not in st.session_state: st.session_state[k]=None
if "theme" not in st.session_state: st.session_state.theme="dark"

T=st.session_state.theme
if T=="dark":
    BG="#000000";BG2="#0d0d0d";BG3="#141414";BG4="#1c1c1c"
    BORDER="#222222";TEXT1="#f9fafb";TEXT2="#9ca3af";TEXT3="#6b7280"
    ACCENT1="#4ade80";ACCENT2="#60a5fa";ACCENT3="#c084fc"
    ACCENTR="#f87171";ACCENTY="#fbbf24"
    HDR_BG="linear-gradient(135deg,#000000 0%,#0d0d0d 60%,#000000 100%)"
    HDR_BORDER="rgba(74,222,128,0.25)"
    BTN_BG="linear-gradient(135deg,#16a34a,#22c55e)";BTN_GLOW="rgba(74,222,128,0.40)"
    TAB_SEL="linear-gradient(135deg,#16a34a,#22c55e)"
    CARD_BG="#0d0d0d";CHART_TEMPLATE="plotly_dark"
    CHART_PAPER="rgba(0,0,0,0)";CHART_FONT="#9ca3af";CHART_GRID="#1c1c1c"
    GLOW_DIV="linear-gradient(90deg,transparent,#4ade80,#60a5fa,transparent)"
    HERO_H1_GRAD="linear-gradient(135deg,#4ade80 0%,#60a5fa 50%,#c084fc 100%)"
else:
    BG="#f8f4ff";BG2="#ffffff";BG3="#ede9fe";BG4="#ddd6fe"
    BORDER="#c4b5fd";TEXT1="#1e0a3c";TEXT2="#5b21b6";TEXT3="#7c3aed"
    ACCENT1="#7c3aed";ACCENT2="#2563eb";ACCENT3="#0891b2"
    ACCENTR="#dc2626";ACCENTY="#d97706"
    HDR_BG="linear-gradient(135deg,#1e0a3c 0%,#4c1d95 50%,#2e1065 100%)"
    HDR_BORDER="rgba(167,139,250,0.4)"
    BTN_BG="linear-gradient(135deg,#5b21b6,#7c3aed)";BTN_GLOW="rgba(124,58,237,0.40)"
    TAB_SEL="linear-gradient(135deg,#5b21b6,#7c3aed)"
    CARD_BG="#ffffff";CHART_TEMPLATE="plotly_white"
    CHART_PAPER="rgba(0,0,0,0)";CHART_FONT="#5b21b6";CHART_GRID="#ddd6fe"
    GLOW_DIV="linear-gradient(90deg,transparent,#7c3aed,#2563eb,transparent)"
    HERO_H1_GRAD="linear-gradient(135deg,#7c3aed 0%,#2563eb 50%,#0891b2 100%)"

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
section[data-testid="stSidebar"] .stButton>button{{background:{"linear-gradient(135deg,#16a34a,#22c55e)" if T=="dark" else "linear-gradient(135deg,#5b21b6,#7c3aed)"} !important;color:#ffffff !important;}}
[data-testid="stFileUploader"]{{background:{"#0f0f0f" if T=="dark" else "#f3eeff"} !important;border-radius:14px !important;}}
[data-testid="stFileUploader"]>div{{background:{"#0d0d0d" if T=="dark" else "#ede9fe"} !important;border:2px dashed {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(124,58,237,0.40)"} !important;border-radius:12px !important;}}
[data-testid="stFileUploader"] *{{color:{"#4ade80" if T=="dark" else "#7c3aed"} !important;background:transparent !important;}}
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
.stButton>button{{background:{BTN_BG} !important;color:#fff !important;border:none !important;padding:.8rem 1.75rem !important;font-weight:700 !important;font-size:.95rem !important;border-radius:12px !important;box-shadow:0 4px 16px {BTN_GLOW} !important;}}
.stButton>button:hover{{transform:translateY(-2px) !important;filter:brightness(1.08) !important;}}
.stDownloadButton>button{{background:{BG3} !important;color:{TEXT1} !important;border:1px solid {BORDER} !important;border-radius:12px !important;font-weight:600 !important;}}
div[data-testid="column"]:nth-child(2) .stButton>button{{background:{BG2 if T=="dark" else "#ffffff"} !important;color:{TEXT1} !important;border:1px solid {BORDER} !important;font-weight:600 !important;}}
div[data-testid="column"]:nth-child(3) .stButton>button{{background:{BG2 if T=="dark" else "#ffffff"} !important;color:{TEXT1} !important;border:1px solid {BORDER} !important;font-weight:600 !important;}}
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{BG};}}
::-webkit-scrollbar-thumb{{background:{BG4};border-radius:3px;}}
::-webkit-scrollbar-thumb:hover{{background:{ACCENT1};}}
.dataframe{{font-family:'JetBrains Mono',monospace !important;font-size:.82rem !important;}}
.stAlert{{border-radius:12px !important;border-left-width:4px !important;}}
@keyframes slideUp{{from{{opacity:0;transform:translateY(18px)}}to{{opacity:1;transform:none}}}}
.slide-up{{animation:slideUp .45s ease-out both;}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def detect_problem_type(s):
    if s.dtype=="object" or str(s.dtype)=="category": return "classification"
    if s.dtype=="bool": return "classification"
    u,n=s.nunique(),len(s)
    if u<=10 and pd.api.types.is_integer_dtype(s): return "classification"
    if u/n<0.05 and u<=20: return "classification"
    return "regression"

def fmt_time(s): return f"{int(s//60)}m {int(s%60)}s" if s>=60 else f"{s:.1f}s"

def chart_layout(**kwargs):
    base=dict(template=CHART_TEMPLATE,paper_bgcolor=CHART_PAPER,plot_bgcolor=CHART_PAPER,
              font=dict(family="Inter",color=CHART_FONT,size=11),
              margin=dict(t=44,b=20,l=20,r=20),title_font=dict(size=13,color=TEXT1))
    base.update(kwargs); return base


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="vibe-header slide-up">
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
""", unsafe_allow_html=True)

_tcol1,_tcol2=st.columns([10,1])
with _tcol2:
    if st.button("⬜ White" if T=="dark" else "⬛ Black",key="theme_btn"):
        st.session_state.theme="light" if T=="dark" else "dark"; st.rerun()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style="background:{"rgba(74,222,128,0.08)" if T=="dark" else "#ffffff"};border:1px solid {"rgba(74,222,128,0.30)" if T=="dark" else "#dddddd"};border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem;text-align:center">
      <div style="font-size:1.5rem;margin-bottom:.3rem">🎁</div>
      <div style="font-size:.85rem;font-weight:800;color:{"#4ade80" if T=="dark" else "#7c3aed"}">All Features Free</div>
      <div style="font-size:.72rem;color:{"#9ca3af" if T=="dark" else "#888"};margin-top:.2rem">XGBoost · LightGBM · CatBoost · 10-fold CV · Model Export</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-title">📂 Data Source</div>', unsafe_allow_html=True)
    uploaded=st.file_uploader("Upload CSV / Excel",type=["csv","xlsx","xls"],label_visibility="collapsed")
    if uploaded:
        try:
            df_up=pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            if st.session_state.get("_last_uploaded")!=uploaded.name:
                st.session_state.data=df_up; st.session_state.dataset_name=uploaded.name
                st.session_state.results=None; st.session_state.best_model=None
                st.session_state["_last_uploaded"]=uploaded.name
                st.session_state.pop("sample_hint",None); st.rerun()
        except Exception as e: st.error(str(e))

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-title">🎯 Sample Datasets</div>', unsafe_allow_html=True)
    sample=st.selectbox("Pick one",["— choose —","🚢 Titanic","💎 Diamonds","🌸 Iris"],label_visibility="collapsed",key="sample_dataset_select",index=0)
    if sample!="— choose —":
        if st.button("Load Sample →",key="load_sample"):
            try:
                urls={"🚢 Titanic":("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv","Survived"),"💎 Diamonds":("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv","price"),"🌸 Iris":("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv","species")}
                url,hint=urls[sample]; df_s=pd.read_csv(url)
                st.session_state.data=df_s; st.session_state.dataset_name=sample
                st.session_state.results=None; st.session_state.best_model=None
                st.session_state["sample_hint"]=hint; st.rerun()
            except Exception as e: st.error(str(e))
    if st.session_state.get("sample_hint"): st.info(f"💡 Target: **{st.session_state['sample_hint']}**")

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
    if st.session_state.data is not None:
        df_sb=st.session_state.data
        null_pct_sb=round(df_sb.isnull().sum().sum()/df_sb.size*100,2)
        hs=round(max(0,100-null_pct_sb*2-df_sb.duplicated().sum()/len(df_sb)*300),1)
        hc=ACCENT1 if hs>80 else ACCENTY if hs>50 else ACCENTR
        st.markdown(f"""<div class="sidebar-section">
          <div class="sidebar-title">📊 Dataset Health</div>
          <div style="font-size:2.2rem;font-weight:900;color:{hc}">{hs}<span style="font-size:1rem;color:{TEXT3}">/100</span></div>
          <div style="font-size:.75rem;color:{TEXT3};margin-top:.3rem">{df_sb.isnull().sum().sum()} nulls · {df_sb.duplicated().sum()} duplicates</div>
          <div style="height:6px;background:{'#1c1c1c' if T=='dark' else '#e0e0e0'};border-radius:3px;margin-top:.75rem;overflow:hidden">
            <div style="height:100%;width:{hs}%;background:linear-gradient(90deg,{hc},{hc}99);border-radius:3px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-title">⚡ Stack</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-section"><span class="insight-chip">PyCaret</span><span class="insight-chip">Plotly</span><span class="insight-chip">Pandas</span><span class="insight-chip">Streamlit</span></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────
if st.session_state.data is not None:
    df=st.session_state.data
    null_pct=round(df.isnull().sum().sum()/df.size*100,2)
    dup_cnt=df.duplicated().sum()
    num_cols=df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols=df.select_dtypes(include=["object","category"]).columns.tolist()

    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["📊  Data Explorer","🧬  EDA & Insights","⚙️  Train Model","🏆  Results","📜  History","📓  Notebook Builder"])

    # ═══════════════════════════
    # TAB 1 — DATA EXPLORER
    # ═══════════════════════════
    with tab1:
        st.markdown(f"""<div class="stat-grid slide-up">
          <div class="stat-card {'good' if null_pct<=5 else 'warn' if null_pct<=15 else 'danger'}"><div class="bar"></div><div class="label">Total Rows</div><div class="value">{len(df):,}</div><div class="sub">records</div></div>
          <div class="stat-card"><div class="bar"></div><div class="label">Columns</div><div class="value">{len(df.columns)}</div><div class="sub">features</div></div>
          <div class="stat-card"><div class="bar"></div><div class="label">Numerical</div><div class="value">{len(num_cols)}</div><div class="sub">numeric cols</div></div>
          <div class="stat-card"><div class="bar"></div><div class="label">Categorical</div><div class="value">{len(cat_cols)}</div><div class="sub">text cols</div></div>
          <div class="stat-card {'good' if null_pct==0 else 'warn' if null_pct<=10 else 'danger'}"><div class="bar"></div><div class="label">Missing</div><div class="value">{null_pct}%</div><div class="sub">{df.isnull().sum().sum()} cells</div></div>
        </div>""", unsafe_allow_html=True)
        if dup_cnt>0: st.warning(f"⚠️ **{dup_cnt}** duplicate rows found.")
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">⚡</div><h3>Quick Actions</h3></div>""", unsafe_allow_html=True)
        qa1,qa2,qa3,qa4=st.columns(4)
        with qa1:
            if st.button("🗑️ Drop Duplicates",key="drop_dups"):
                before=len(df); st.session_state.data=df.drop_duplicates().reset_index(drop=True)
                st.success(f"Removed {before-len(st.session_state.data)} duplicates!"); st.rerun()
        with qa2:
            if st.button("🧹 Drop All-Null Cols",key="drop_null_cols"):
                before=len(df.columns); st.session_state.data=df.dropna(axis=1,how='all')
                st.success(f"Removed {before-len(st.session_state.data.columns)} empty columns!"); st.rerun()
        with qa3:
            if st.button("📊 Show Data Types",key="show_dtypes"):
                st.dataframe(df.dtypes.reset_index().rename(columns={"index":"Column",0:"Type"}))
        with qa4:
            st.download_button("📥 Download CSV",df.to_csv(index=False),f"dataset_{pd.Timestamp.now().strftime('%Y%m%d')}.csv","text/csv")
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">🔍</div><h3>Data Preview</h3></div>""", unsafe_allow_html=True)
        search=st.text_input("Filter columns (comma-separated)",placeholder="e.g. Age, Sex, Survived",label_visibility="collapsed")
        show_rows=st.slider("Rows to show",5,100,20,key="preview_rows")
        if search.strip():
            cols_f=[c.strip() for c in search.split(",") if c.strip() in df.columns]
            st.dataframe((df[cols_f] if cols_f else df).head(show_rows))
        else: st.dataframe(df.head(show_rows))
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📋</div><h3>Column Details</h3></div>""", unsafe_allow_html=True)
            ci=pd.DataFrame({"Column":df.columns,"Type":df.dtypes.astype(str).values,"Non-Null":df.count().values,"Null %":((df.isnull().sum()/len(df))*100).round(1).astype(str)+"%","Unique":df.nunique().values})
            st.dataframe(ci.astype(str),height=300)
        with c2:
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📊</div><h3>Statistical Summary</h3></div>""", unsafe_allow_html=True)
            st.dataframe(df.describe().round(3),height=300)

    # ═══════════════════════════
    # TAB 2 — EDA
    # ═══════════════════════════
    with tab2:
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">🧬</div><h3>Exploratory Data Analysis</h3></div>""", unsafe_allow_html=True)
        eda_sub1,eda_sub2,eda_sub3,eda_sub4,eda_sub5,eda_sub6=st.tabs(["📈 Distributions","🔥 Correlations","⚠️ Outliers","🎯 Target Analysis","🔤 Categorical","📐 Advanced Stats"])

        with eda_sub1:
            if not num_cols: st.info("No numerical columns found.")
            else:
                st.markdown("#### 📈 Distribution Explorer")
                dv1,dv2=st.columns([3,2])
                with dv1: col_pick=st.selectbox("Select column",num_cols+cat_cols,key="eda_col")
                with dv2: chart_type=st.selectbox("Chart type",["Histogram","Box","Violin"] if col_pick in num_cols else ["Bar Chart"],key="chart_type")
                cv1,cv2=st.columns([3,2])
                with cv1:
                    if col_pick in num_cols:
                        if chart_type=="Histogram": fig=px.histogram(df,x=col_pick,nbins=40,color_discrete_sequence=[ACCENT1],template=CHART_TEMPLATE,title=f"Distribution · {col_pick}")
                        elif chart_type=="Box": fig=px.box(df,y=col_pick,color_discrete_sequence=[ACCENT1],template=CHART_TEMPLATE,title=f"Box Plot · {col_pick}")
                        else: fig=px.violin(df,y=col_pick,color_discrete_sequence=[ACCENT1],box=True,template=CHART_TEMPLATE,title=f"Violin · {col_pick}")
                    else:
                        vc=df[col_pick].value_counts().head(15)
                        fig=px.bar(x=vc.index,y=vc.values,color=vc.values,color_continuous_scale=[ACCENT2,ACCENT1],template=CHART_TEMPLATE,title=f"Top values · {col_pick}")
                        fig.update_layout(showlegend=False,coloraxis_showscale=False)
                    fig.update_layout(**chart_layout(height=340)); st.plotly_chart(fig,use_container_width=True)
                with cv2:
                    s=df[col_pick]; rows_dict={"Count":f"{s.count():,}","Missing":f"{s.isnull().sum()} ({s.isnull().mean()*100:.1f}%)","Unique":f"{s.nunique():,}"}
                    if col_pick in num_cols:
                        from scipy.stats import skew as _skew,kurtosis as _kurt
                        rows_dict.update({"Mean":f"{s.mean():.4f}","Std":f"{s.std():.4f}","Min":f"{s.min():.4f}","Median":f"{s.median():.4f}","Max":f"{s.max():.4f}","Skewness":f"{_skew(s.dropna()):.3f}","Kurtosis":f"{_kurt(s.dropna()):.3f}"})
                    st.markdown(f'<div class="sidebar-section" style="margin-top:2.2rem">', unsafe_allow_html=True)
                    for k,v in rows_dict.items():
                        st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid {BORDER}"><span style="font-size:.78rem;color:{TEXT3};font-weight:600">{k}</span><span style="font-size:.82rem;color:{TEXT1};font-family:'JetBrains Mono',monospace">{v}</span></div>""", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                if len(num_cols)>1:
                    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                    st.markdown("#### 📊 All Numeric Distributions Grid")
                    max_plot_cols=st.slider("Max features",3,min(12,len(num_cols)),min(9,len(num_cols)),key="dist_grid_slider")
                    cols_to_plot=num_cols[:max_plot_cols]; n_c=min(3,len(cols_to_plot)); n_r=(len(cols_to_plot)+n_c-1)//n_c
                    from plotly.subplots import make_subplots
                    fig_grid=make_subplots(rows=n_r,cols=n_c,subplot_titles=cols_to_plot)
                    for i,col in enumerate(cols_to_plot):
                        fig_grid.add_trace(go.Histogram(x=df[col],name=col,marker_color=ACCENT1,opacity=0.8),row=i//n_c+1,col=i%n_c+1)
                    fig_grid.update_layout(**chart_layout(height=280*n_r,title_text="📊 All Feature Distributions"),showlegend=False)
                    st.plotly_chart(fig_grid,use_container_width=True)

        with eda_sub2:
            if len(num_cols)>=2:
                st.markdown("#### 🔥 Correlation Heatmap")
                corr_cols=st.multiselect("Select columns",num_cols,default=num_cols[:min(12,len(num_cols))],key="corr_cols_select")
                if len(corr_cols)>=2:
                    corr=df[corr_cols].corr().round(3)
                    fig_h=go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,colorscale=[[0,ACCENTR],[.5,BG2],[1,ACCENT1]],zmid=0,text=corr.values.round(2),texttemplate="%{text}",textfont=dict(size=9,family="JetBrains Mono")))
                    fig_h.update_layout(**chart_layout(height=500)); st.plotly_chart(fig_h,use_container_width=True)
                    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                    st.markdown("#### 🔗 Strongest Correlated Pairs")
                    corr_pairs=[]
                    for i in range(len(corr.columns)):
                        for j in range(i+1,len(corr.columns)):
                            corr_pairs.append({"Feature A":corr.columns[i],"Feature B":corr.columns[j],"Correlation":round(corr.iloc[i,j],4),"Abs":abs(round(corr.iloc[i,j],4))})
                    if corr_pairs:
                        pairs_df=pd.DataFrame(corr_pairs).sort_values("Abs",ascending=False).drop("Abs",axis=1)
                        top_pairs=pairs_df.head(15); colors_p=[ACCENTR if v<0 else ACCENT1 for v in top_pairs["Correlation"]]
                        fig_p=go.Figure(go.Bar(x=top_pairs["Correlation"],y=[f"{a} ↔ {b}" for a,b in zip(top_pairs["Feature A"],top_pairs["Feature B"])],orientation="h",marker_color=colors_p,text=[f"{v:.3f}" for v in top_pairs["Correlation"]],textposition="outside"))
                        fig_p.update_layout(**chart_layout(height=max(300,len(top_pairs)*28),title="🔗 Top 15 Correlated Pairs")); st.plotly_chart(fig_p,use_container_width=True)
                    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                    st.markdown("#### 🔵 Feature vs Feature Scatter")
                    sc1,sc2,sc3=st.columns(3)
                    with sc1: x_feat=st.selectbox("X-axis",num_cols,key="scatter_x")
                    with sc2: y_feat=st.selectbox("Y-axis",num_cols,index=min(1,len(num_cols)-1),key="scatter_y")
                    with sc3: color_feat=st.selectbox("Color by",["None"]+cat_cols+num_cols,key="scatter_color")
                    color_col=None if color_feat=="None" else color_feat
                    fig_sc=px.scatter(df,x=x_feat,y=y_feat,color=color_col,title=f"🔵 {x_feat} vs {y_feat}",trendline="ols",template=CHART_TEMPLATE,opacity=0.7,color_discrete_sequence=[ACCENT1,ACCENT2,ACCENT3,ACCENTR])
                    fig_sc.update_layout(**chart_layout(height=420)); st.plotly_chart(fig_sc,use_container_width=True)
            else: st.info("Need at least 2 numerical columns.")

        with eda_sub3:
            if not num_cols: st.info("No numerical columns found.")
            else:
                st.markdown("#### ⚠️ Outlier Detection — IQR Method")
                outlier_data=[]
                for col in num_cols:
                    s=df[col].dropna(); Q1,Q3=s.quantile(0.25),s.quantile(0.75); IQR=Q3-Q1
                    lower=Q1-1.5*IQR; upper=Q3+1.5*IQR; n_low=int((s<lower).sum()); n_high=int((s>upper).sum())
                    outlier_data.append({"Feature":col,"Q1":round(Q1,3),"Q3":round(Q3,3),"IQR":round(IQR,3),"Lower Fence":round(lower,3),"Upper Fence":round(upper,3),"Low Outliers":n_low,"High Outliers":n_high,"Total Outliers":n_low+n_high,"Outlier%":round((n_low+n_high)/len(df)*100,2)})
                out_df=pd.DataFrame(outlier_data).sort_values("Total Outliers",ascending=False)
                total_out_rows=int(df[num_cols].apply(lambda col:((col<col.quantile(0.25)-1.5*(col.quantile(0.75)-col.quantile(0.25)))|(col>col.quantile(0.75)+1.5*(col.quantile(0.75)-col.quantile(0.25))))).any(axis=1).sum())
                clean_pct=round((1-total_out_rows/len(df))*100,1)
                o1,o2,o3,o4=st.columns(4)
                for col_w,label,val,sub,cls in [(o1,"Outlier Rows",f"{total_out_rows:,}",f"{100-clean_pct:.1f}% of data","warn" if total_out_rows>0 else "good"),(o2,"Clean Rows",f"{len(df)-total_out_rows:,}",f"{clean_pct:.1f}% of data","good"),(o3,"Features w/ Outliers",f"{int((out_df['Total Outliers']>0).sum())}",f"of {len(num_cols)} numeric","warn"),(o4,"Most Outliers",out_df.iloc[0]["Feature"] if len(out_df)>0 else "—",f"{out_df.iloc[0]['Outlier%']:.1f}%" if len(out_df)>0 else "","danger")]:
                    with col_w:
                        st.markdown(f"""<div class="stat-card {cls}"><div class="bar"></div><div class="label">{label}</div><div class="value" style="font-size:1.4rem">{val}</div><div class="sub">{sub}</div></div>""", unsafe_allow_html=True)
                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                fig_out=go.Figure()
                fig_out.add_trace(go.Bar(x=out_df["Feature"],y=out_df["Low Outliers"],name="Low Outliers",marker_color=ACCENT2))
                fig_out.add_trace(go.Bar(x=out_df["Feature"],y=out_df["High Outliers"],name="High Outliers",marker_color=ACCENTR))
                fig_out.update_layout(**chart_layout(barmode="stack",height=380,title="⚠️ Low vs High Outliers per Feature")); st.plotly_chart(fig_out,use_container_width=True)
                out_sorted=out_df.sort_values("Outlier%",ascending=True)
                colors_out=[ACCENTR if v>5 else ACCENTY if v>2 else ACCENT1 for v in out_sorted["Outlier%"]]
                fig_out2=go.Figure(go.Bar(x=out_sorted["Outlier%"],y=out_sorted["Feature"],orientation="h",marker_color=colors_out,text=[f"{v:.1f}%" for v in out_sorted["Outlier%"]],textposition="outside"))
                fig_out2.update_layout(**chart_layout(height=max(280,len(out_sorted)*30),title="📊 Outlier % per Feature")); st.plotly_chart(fig_out2,use_container_width=True)
                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown("#### 📋 Detailed IQR Table")
                st.dataframe(out_df.style.background_gradient(subset=["Outlier%"],cmap="RdYlGn_r").format({"Outlier%":"{:.2f}%"}),height=300)
                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown("#### 🔍 Outlier Rows Inspector")
                inspect_col=st.selectbox("Inspect outliers in column",num_cols,key="inspect_col")
                s_ins=df[inspect_col]; Q1_i,Q3_i=s_ins.quantile(0.25),s_ins.quantile(0.75); IQR_i=Q3_i-Q1_i
                mask_i=(s_ins<Q1_i-1.5*IQR_i)|(s_ins>Q3_i+1.5*IQR_i); outlier_rows_df=df[mask_i]
                st.info(f"**{inspect_col}** — Lower fence: `{Q1_i-1.5*IQR_i:.4f}` | Upper fence: `{Q3_i+1.5*IQR_i:.4f}` | Outlier rows: **{len(outlier_rows_df):,}**")
                if len(outlier_rows_df)>0: st.dataframe(outlier_rows_df.head(20),height=250)
                else: st.success(f"✅ No outliers found in {inspect_col}!")

        with eda_sub4:
            st.markdown("#### 🎯 Target Variable Analysis")
            target_sel=st.selectbox("Select target column",df.columns.tolist(),key="eda_target_sel")
            if target_sel:
                ts=df[target_sel]; ptype_eda=detect_problem_type(ts)
                st.info(f"**{target_sel}** — Detected: **{'🎯 Classification' if ptype_eda=='classification' else '📈 Regression'}** | Unique values: {ts.nunique():,} | Missing: {ts.isnull().sum()} ({ts.isnull().mean()*100:.1f}%)")
                ta1,ta2=st.columns(2)
                with ta1:
                    if ptype_eda=="classification":
                        vc=ts.value_counts()
                        fig_tgt=px.pie(values=vc.values,names=vc.index.astype(str),title=f"🎯 {target_sel} Class Distribution",color_discrete_sequence=[ACCENT1,ACCENT2,ACCENT3,ACCENTR,ACCENTY])
                    else: fig_tgt=px.histogram(df,x=target_sel,nbins=50,title=f"📈 {target_sel} Distribution",color_discrete_sequence=[ACCENT1])
                    fig_tgt.update_layout(**chart_layout(height=360)); st.plotly_chart(fig_tgt,use_container_width=True)
                with ta2:
                    if ptype_eda=="classification":
                        vc=ts.value_counts()
                        fig_bar_tgt=px.bar(x=vc.index.astype(str),y=vc.values,title=f"📊 {target_sel} Class Counts",color=vc.values,color_continuous_scale=[ACCENT2,ACCENT1])
                        fig_bar_tgt.update_layout(**chart_layout(height=360),coloraxis_showscale=False); st.plotly_chart(fig_bar_tgt,use_container_width=True)
                    else:
                        fig_box_tgt=px.box(df,y=target_sel,title=f"📦 {target_sel} Box Plot",color_discrete_sequence=[ACCENT1])
                        fig_box_tgt.update_layout(**chart_layout(height=360)); st.plotly_chart(fig_box_tgt,use_container_width=True)

        with eda_sub5:
            if not cat_cols: st.info("No categorical columns found.")
            else:
                st.markdown("#### 🔤 Categorical Features Analysis")
                cat_summary_rows=[]
                for col in cat_cols:
                    s=df[col]; vc=s.value_counts()
                    cat_summary_rows.append({"Feature":col,"Unique":s.nunique(),"Missing%":f"{s.isnull().mean()*100:.1f}%","Top Value":str(vc.index[0]) if len(vc)>0 else "N/A","Top Count":int(vc.iloc[0]) if len(vc)>0 else 0,"Top%":f"{vc.iloc[0]/len(df)*100:.1f}%" if len(vc)>0 else "0%","Cardinality":"🟢 Low" if s.nunique()<=5 else "🟡 Medium" if s.nunique()<=20 else "🔴 High"})
                cat_sum_df=pd.DataFrame(cat_summary_rows); st.dataframe(cat_sum_df,height=200)
                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                fig_card=px.bar(cat_sum_df.sort_values("Unique",ascending=True),x="Unique",y="Feature",orientation="h",title="🔤 Cardinality per Categorical Feature",color="Unique",color_continuous_scale=[ACCENT1,ACCENTY,ACCENTR])
                fig_card.update_layout(**chart_layout(height=max(280,len(cat_cols)*35)),coloraxis_showscale=False); st.plotly_chart(fig_card,use_container_width=True)
                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown("#### 📋 Value Counts Detail")
                cat_detail_col=st.selectbox("Select categorical column",cat_cols,key="cat_detail_sel")
                top_n=st.slider("Top N values",5,30,15,key="cat_top_n")
                vc_detail=df[cat_detail_col].value_counts().head(top_n)
                cd1,cd2=st.columns(2)
                with cd1:
                    fig_pie_cat=px.pie(values=vc_detail.values,names=vc_detail.index.astype(str),title=f"🥧 {cat_detail_col}",color_discrete_sequence=[ACCENT1,ACCENT2,ACCENT3,ACCENTR,ACCENTY,"#a78bfa","#fb7185","#34d399"])
                    fig_pie_cat.update_layout(**chart_layout(height=380)); st.plotly_chart(fig_pie_cat,use_container_width=True)
                with cd2:
                    fig_bar_cat=px.bar(x=vc_detail.index.astype(str),y=vc_detail.values,title=f"📊 {cat_detail_col}",color=vc_detail.values,color_continuous_scale=[ACCENT2,ACCENT1])
                    fig_bar_cat.update_layout(**chart_layout(height=380),coloraxis_showscale=False); st.plotly_chart(fig_bar_cat,use_container_width=True)

        with eda_sub6:
            if not num_cols: st.info("No numerical columns.")
            else:
                st.markdown("#### 📐 Skewness & Kurtosis Analysis")
                try:
                    from scipy.stats import skew as _skew2,kurtosis as _kurt2
                    sk_rows=[]
                    for col in num_cols:
                        s=df[col].dropna()
                        if len(s)>3:
                            sk_rows.append({"Feature":col,"Skewness":round(_skew2(s),3),"Kurtosis":round(_kurt2(s),3),"Mean":round(s.mean(),4),"Std Dev":round(s.std(),4),"Normality":"✅ ~Normal" if abs(_skew2(s))<0.5 else "⚠️ Skewed" if abs(_skew2(s))<1 else "🔴 Highly Skewed"})
                    sk_df=pd.DataFrame(sk_rows).sort_values("Skewness",key=abs,ascending=False)
                    st.dataframe(sk_df,height=280)
                    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                    from plotly.subplots import make_subplots as _msp
                    fig_sk=_msp(rows=1,cols=2,subplot_titles=["Skewness per Feature","Kurtosis per Feature"])
                    colors_sk2=[ACCENTR if abs(v)>1 else ACCENTY if abs(v)>0.5 else ACCENT1 for v in sk_df["Skewness"]]
                    fig_sk.add_trace(go.Bar(x=sk_df["Skewness"],y=sk_df["Feature"],orientation="h",marker_color=colors_sk2,name="Skewness"),row=1,col=1)
                    fig_sk.add_trace(go.Bar(x=sk_df["Kurtosis"],y=sk_df["Feature"],orientation="h",marker_color=ACCENT3,name="Kurtosis"),row=1,col=2)
                    fig_sk.update_layout(**chart_layout(height=max(350,len(sk_df)*32),title_text="📐 Skewness & Kurtosis"),showlegend=False); st.plotly_chart(fig_sk,use_container_width=True)
                    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                    st.markdown("#### 📊 Enhanced Numeric Summary")
                    enh_rows=[]
                    for col in num_cols:
                        s=df[col].dropna(); Q1,Q3=s.quantile(0.25),s.quantile(0.75); IQR=Q3-Q1; n_out=int(((s<Q1-1.5*IQR)|(s>Q3+1.5*IQR)).sum())
                        enh_rows.append({"Feature":col,"Count":int(s.count()),"Missing%":round(df[col].isnull().mean()*100,1),"Mean":round(s.mean(),4),"Std":round(s.std(),4),"Min":round(s.min(),4),"Q1":round(Q1,4),"Median":round(s.median(),4),"Q3":round(Q3,4),"Max":round(s.max(),4),"IQR":round(IQR,4),"Outliers":n_out,"Outlier%":round(n_out/len(df)*100,1)})
                    enh_df=pd.DataFrame(enh_rows)
                    st.dataframe(enh_df.style.background_gradient(subset=["Missing%","Outlier%"],cmap="RdYlGn_r").format(precision=3),height=320)
                    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                    st.markdown("#### 🔔 Quick Feature Importance (Pre-Training)")
                    fi_target=st.selectbox("Select target column",df.columns.tolist(),key="fi_eda_target")
                    if fi_target and st.button("⭐ Compute Quick Importance",key="fi_eda_btn"):
                        with st.spinner("Running RandomForest..."):
                            try:
                                from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
                                from sklearn.preprocessing import LabelEncoder
                                X_q=df.select_dtypes(include="number").drop(columns=[fi_target],errors="ignore").fillna(0)
                                y_q=df[fi_target].fillna(df[fi_target].mode()[0]); ptype_fi=detect_problem_type(y_q)
                                if ptype_fi=="classification":
                                    le=LabelEncoder(); y_enc=le.fit_transform(y_q.astype(str))
                                    rf_fi=RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=-1,max_depth=5); rf_fi.fit(X_q,y_enc)
                                else:
                                    rf_fi=RandomForestRegressor(n_estimators=50,random_state=42,n_jobs=-1,max_depth=5); rf_fi.fit(X_q,y_q)
                                fi_df2=pd.DataFrame({"Feature":X_q.columns,"Importance":rf_fi.feature_importances_}).sort_values("Importance",ascending=True)
                                fig_fi=px.bar(fi_df2,x="Importance",y="Feature",orientation="h",title=f"⭐ Feature Importance for `{fi_target}`",color="Importance",color_continuous_scale=[ACCENT2,ACCENT1,ACCENTY],text=fi_df2["Importance"].round(4))
                                fig_fi.update_layout(**chart_layout(height=max(300,len(fi_df2)*30)),coloraxis_showscale=False); fig_fi.update_traces(textposition="outside"); st.plotly_chart(fig_fi,use_container_width=True)
                            except Exception as e: st.error(f"Could not compute: {e}")
                except Exception as e: st.error(f"Error: {e}")

    # ═══════════════════════════
    # TAB 3 — TRAIN MODEL
    # ═══════════════════════════
    with tab3:
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">⚙️</div><h3>Training Configuration</h3></div>""", unsafe_allow_html=True)
        tc1,tc2=st.columns([3,1])
        with tc1: target_col=st.selectbox("🎯 Select Target Column",df.columns.tolist())
        with tc2:
            st.markdown("<br>", unsafe_allow_html=True)
            row_color=ACCENT1 if len(df)<=MAX_ROWS_WARNING else ACCENTY if len(df)<=MAX_ROWS_TRAINING else ACCENTR
            st.markdown(f'<div style="padding:.6rem 1rem;background:{BG3};border:1px solid {row_color}44;border-radius:10px;text-align:center"><div style="font-size:.65rem;color:{TEXT3};text-transform:uppercase;font-weight:700">Dataset Size</div><div style="font-size:1.1rem;font-weight:900;color:{row_color}">{len(df):,}</div><div style="font-size:.62rem;color:{TEXT3}">rows</div></div>', unsafe_allow_html=True)
        if target_col:
            ts=df[target_col]; ptype=detect_problem_type(ts); uniq=ts.nunique()
            st.session_state.problem_type=ptype
            card_cls="clf" if ptype=="classification" else "reg"
            icon="🎯" if ptype=="classification" else "📈"
            type_lbl="Classification" if ptype=="classification" else "Regression"
            st.markdown(f"""<div class="target-card {card_cls} slide-up"><div class="tc-icon">{icon}</div><div><div class="tc-label">Problem Type Detected</div><div class="tc-type">{type_lbl}</div><div class="tc-meta">Target: <code>{target_col}</code> · {uniq} unique values · Auto CV enabled</div></div></div>""", unsafe_allow_html=True)
            if len(df)>MAX_ROWS_TRAINING: st.error(f"🚨 **Dataset {len(df):,} rows** — will auto-sample **{MAX_ROWS_TRAINING:,} rows**.")
            elif len(df)>MAX_ROWS_WARNING: st.warning(f"⚠️ **{len(df):,} rows** — training will proceed.")
            available_models=ALL_CLF_MODELS if ptype=="classification" else ALL_REG_MODELS
            st.markdown(f"""<div style="background:linear-gradient(135deg,rgba(74,222,128,0.12),rgba(96,165,250,0.08));border:1.5px solid rgba(74,222,128,0.4);border-radius:14px;padding:.9rem 1.25rem;margin:.75rem 0;display:flex;align-items:center;gap:1rem"><div style="font-size:2rem">🎁</div><div><div style="font-size:.9rem;font-weight:900;color:#4ade80;">ALL FEATURES UNLOCKED — 100% FREE</div><div style="font-size:.72rem;color:#9ca3af;margin-top:.15rem">XGBoost ✅ LightGBM ✅ CatBoost ✅ 10-fold CV ✅ Unlimited ✅ Export ✅</div></div></div>""", unsafe_allow_html=True)
            with st.expander("⚙️ Advanced Configuration",expanded=False):
                ac1,ac2,ac3=st.columns(3)
                with ac1: train_size=st.slider("Training Split",0.5,0.9,0.8,0.05)
                with ac2:
                    recommended_fold=min(3,10) if len(df)>MAX_ROWS_WARNING else 5
                    fold=st.slider("CV Folds",min_value=2,max_value=10,value=min(recommended_fold,10))
                with ac3: max_models_slider=st.slider(f"Max Models",min_value=2,max_value=len(available_models),value=len(available_models))
                ac4,ac5=st.columns(2)
                with ac4: normalize=st.checkbox("Normalize Features",value=True)
                with ac5: remove_out=st.checkbox("Remove Outliers",value=False)
            st.session_state.cv_fold=fold
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn1,col_btn2=st.columns([3,1])
            with col_btn1: train_clicked=st.button("🚀 Launch Training",key="train_btn")
            with col_btn2:
                if st.session_state.results is not None:
                    if st.button("🔄 Reset Results",key="reset_btn"):
                        st.session_state.results=None; st.session_state.best_model=None; st.session_state.training_time=None; force_gc(); st.rerun()
            if train_clicked:
                progress_bar=st.progress(0); status_box=st.empty(); warn_box=st.empty(); timeline_box=st.empty()
                steps_labels=["📦 Data Sampling & Validation","⚙️ PyCaret Environment Setup","🤖 Model Comparison","🏆 Best Model Selection","💾 Saving Artifact"]
                def render_steps(done_count):
                    html='<div class="step-timeline">'
                    for i,lbl in enumerate(steps_labels):
                        cls="done" if i<done_count else ("active" if i==done_count else "")
                        icon_s="✓" if i<done_count else ("◉" if i==done_count else str(i+1))
                        html+=f'<div class="step-item {cls}"><div class="step-dot {cls}">{icon_s}</div><div><div class="step-label">{lbl}</div></div></div>'
                    return html+"</div>"
                timeline_box.markdown(render_steps(0),unsafe_allow_html=True); progress_bar.progress(5); status_box.info("🚀 Training is starting...")
                try:
                    best,results,elapsed,warn_msgs,trained_rows=run_memory_safe_training(df=df,target_col=target_col,problem_type=ptype,train_size=train_size,fold=fold,normalize=normalize,remove_out=remove_out,max_models=max_models_slider)
                    progress_bar.progress(100); timeline_box.markdown(render_steps(len(steps_labels)),unsafe_allow_html=True)
                    st.session_state.best_model=best; st.session_state.results=results; st.session_state.training_time=elapsed; st.session_state.target_col=target_col
                    for w in warn_msgs: warn_box.warning(w)
                    if "training_history" not in st.session_state: st.session_state.training_history=[]
                    try:
                        model_col_r="Model" if "Model" in results.columns else results.columns[0]
                        num_res_r=results.select_dtypes(include=[np.number]).columns
                        bm_name_r=str(results.iloc[0][model_col_r]); bm_score_r=float(results.iloc[0][num_res_r[0]]) if len(num_res_r) else 0.0
                        st.session_state.training_history.append({"time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"dataset":str(st.session_state.dataset_name or "Uploaded CSV"),"problem_type":str(ptype),"best_model":bm_name_r,"score":round(bm_score_r,4),"rows":trained_rows,"cols":len(df.columns)})
                    except: pass
                    status_box.success(f"✅ Training complete in **{fmt_time(elapsed)}** ({trained_rows:,} rows) — 🏆 Check the Results tab!"); st.balloons()
                except MemoryError as me: progress_bar.progress(0); timeline_box.empty(); status_box.error(f"💥 **Memory Overflow!**\n{str(me)}")
                except Exception as e: progress_bar.progress(0); timeline_box.empty(); status_box.error(f"❌ Training failed: {str(e)}")

    # ═══════════════════════════
    # TAB 4 — RESULTS
    # ═══════════════════════════
    with tab4:
        if st.session_state.results is None:
            st.markdown(f"""<div style="text-align:center;padding:5rem 2rem"><div style="font-size:5rem;margin-bottom:1rem;opacity:.6">🏆</div><div style="font-size:1.4rem;font-weight:800;color:{TEXT1};margin-bottom:.5rem">No results yet</div><div style="color:{TEXT2}">Train a model in the ⚙️ Train Model tab first</div></div>""", unsafe_allow_html=True)
        else:
            res_df=st.session_state.results; model_col="Model" if "Model" in res_df.columns else res_df.columns[0]
            num_res=res_df.select_dtypes(include=[np.number]).columns.tolist(); best_name=res_df.iloc[0][model_col]
            metric_name=num_res[0] if num_res else "Score"; top_score=res_df.iloc[0][metric_name] if num_res else 0
            folds_used=st.session_state.cv_fold or 5
            st.markdown(f"""<div class="trophy-banner slide-up"><div class="trophy-icon">🏆</div><div class="trophy-text"><h2>{best_name}</h2><p>Best model via {folds_used}-fold cross-validation · Production ready</p></div><div class="trophy-score"><div class="ts-label">{metric_name}</div><div class="ts-value">{top_score:.4f}</div></div></div>""", unsafe_allow_html=True)
            ex1,ex2,ex3=st.columns(3)
            with ex1: st.download_button("📥 Export Results CSV",res_df.to_csv(index=False),f"results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv","text/csv")
            with ex2:
                if os.path.exists("best_model.pkl"):
                    with open("best_model.pkl","rb") as pkl_f: pkl_bytes=pkl_f.read()
                    st.download_button("📦 Download Model (.pkl)",data=pkl_bytes,file_name=f"best_model_{best_name.replace(' ','_')}.pkl",mime="application/octet-stream")
                else: st.info("💾 Model file generates after training.")
            with ex3:
                try:
                    _target_col_nb=st.session_state.get("target_col") or df.columns[0]
                    nb_bytes=generate_notebook(df=df,target_col=_target_col_nb,problem_type=st.session_state.problem_type,results_df=res_df,best_model_name=best_name,top_score=top_score,metric_name=metric_name,dataset_name=str(st.session_state.dataset_name or "dataset"),training_time=st.session_state.training_time or 0,fold=st.session_state.cv_fold or 5,normalize=True,train_size=0.8,eda_selections=st.session_state.get("nb_eda"),pre_selections=st.session_state.get("nb_pre"),train_selections=st.session_state.get("nb_train"),export_selections=st.session_state.get("nb_export"),author_name=st.session_state.get("nb_author_name",""),author_title=st.session_state.get("nb_author_title",""),author_quote=st.session_state.get("nb_author_quote",""),author_email=st.session_state.get("nb_author_email",""),author_linkedin=st.session_state.get("nb_author_linkedin",""),author_github=st.session_state.get("nb_author_github",""),author_kaggle=st.session_state.get("nb_author_kaggle",""),author_facebook=st.session_state.get("nb_author_facebook",""))
                    safe_name=str(st.session_state.dataset_name or "dataset").replace(" ","_").replace(".","_")
                    st.download_button(label="📓 Download Notebook (.ipynb)",data=nb_bytes,file_name=f"dataforge_{safe_name}_{best_name.replace(' ','_')}.ipynb",mime="application/x-ipynb+json")
                except Exception as nb_err: st.error(f"Notebook generation failed: {nb_err}")
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📋</div><h3>All Models Ranked</h3></div>""", unsafe_allow_html=True)
            styled_df=(res_df.style.background_gradient(cmap="RdYlGn",subset=num_res).format({c:"{:.4f}" for c in num_res}).set_properties(**{"font-family":"JetBrains Mono,monospace","font-size":"12px"}))
            st.dataframe(styled_df,height=360)
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            ch1,ch2=st.columns(2)
            with ch1:
                top6=res_df.head(6); colors=[ACCENT1 if i==0 else BG3 for i in range(len(top6))]
                fig_b=go.Figure(go.Bar(x=top6[metric_name],y=top6[model_col],orientation="h",marker_color=colors,text=top6[metric_name].round(4),textposition="inside",textfont=dict(size=10,color="white")))
                fig_b.update_layout(**chart_layout(height=360,title=f"Top Models · {metric_name}",yaxis=dict(autorange="reversed"))); st.plotly_chart(fig_b,use_container_width=True)
            with ch2:
                rc=num_res[:6]; bv=res_df.iloc[0][rc]; mi,ma=bv.min(),bv.max(); nv=(bv-mi)/(ma-mi+1e-9)
                fig_r=go.Figure(go.Scatterpolar(r=list(nv.values)+[nv.values[0]],theta=list(nv.index)+[nv.index[0]],fill="toself",fillcolor="rgba(74,222,128,0.18)",line=dict(color=ACCENT1,width=2.5),marker=dict(size=6,color=ACCENT1)))
                fig_r.update_layout(**chart_layout(height=360,showlegend=False,title="Best Model · Metrics Radar",polar=dict(bgcolor=CHART_PAPER,radialaxis=dict(visible=True,range=[0,1],gridcolor=BORDER),angularaxis=dict(gridcolor=BORDER)))); st.plotly_chart(fig_r,use_container_width=True)

    # ═══════════════════════════
    # TAB 5 — HISTORY
    # ═══════════════════════════
    with tab5:
        training_log=st.session_state.get("training_history",[])
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">🗂️</div><h3>This Session's Training History</h3></div>""", unsafe_allow_html=True)
        st.caption("History is saved for this browser session only.")
        if not training_log:
            st.markdown(f"""<div style="text-align:center;padding:3rem 1rem;background:{CARD_BG};border:1px solid {BORDER};border-radius:20px"><div style="font-size:4rem;margin-bottom:.75rem;opacity:.35">🗂️</div><div style="font-size:1.1rem;font-weight:700;color:{TEXT1}">No projects yet</div><div style="color:{TEXT2};font-size:.875rem">Upload a dataset and train your first model!</div></div>""", unsafe_allow_html=True)
        else:
            for t in training_log[::-1]:
                pt=t.get("problem_type","—"); pt_color=ACCENT1 if pt=="classification" else ACCENT2
                st.markdown(f"""<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;padding:1.25rem;margin-bottom:.75rem;display:flex;align-items:center;gap:1rem;position:relative;overflow:hidden">
                  <div style="position:absolute;top:0;left:0;bottom:0;width:3px;background:{pt_color}"></div>
                  <div style="font-size:1.5rem;margin-left:.5rem">{"🎯" if pt=="classification" else "📈"}</div>
                  <div style="flex:1"><div style="font-size:.9rem;font-weight:700;color:{TEXT1}">{t.get("dataset","?")}</div><div style="font-size:.75rem;color:{TEXT3};margin-top:.15rem">{t.get("best_model","?")} · {t.get("rows",0):,} rows · {t.get("time","")[:16]}</div></div>
                  <div style="text-align:right"><div style="font-size:.65rem;font-weight:800;text-transform:uppercase;color:{TEXT3}">Score</div><div style="font-size:1.3rem;font-weight:900;color:{pt_color};font-family:'JetBrains Mono',monospace">{t.get("score",0):.4f}</div></div>
                </div>""", unsafe_allow_html=True)
            tlog_df=pd.DataFrame(training_log)
            st.download_button("📥 Export History CSV",tlog_df.to_csv(index=False),"training_history.csv","text/csv")

    # ═══════════════════════════════════════════
    # TAB 6 — NOTEBOOK BUILDER PRO (NEW UI)
    # ═══════════════════════════════════════════
    with tab6:

        # ── Session state init ──
        for k,v in [
            ("nb_step",1),
            ("nb_ptype", st.session_state.problem_type or "classification"),
            ("nb_eda",{"distributions":True,"correlation":True,"missing":True,"target_dist":True,"boxplots":False,"scatter_matrix":False,"cat_bars":False,"outlier_plot":False,"violin_plots":True,"skewness_kurtosis":True,"numeric_summary_styled":True,"iqr_analysis":True,"feature_importance_eda":True,"categorical_summary":True,"value_counts_table":False,"outlier_table":True}),
            ("nb_pre",{"drop_dups":True,"handle_missing":True,"normalize":True,"remove_outliers":False}),
            ("nb_train",{"model_table":True,"bar_chart":True,"radar_chart":True,"feature_importance":False}),
            ("nb_export",{"save_model":True,"load_predict":True,"summary_table":True}),
            ("nb_author_name",""),("nb_author_title",""),("nb_author_quote",""),
            ("nb_author_email",""),("nb_author_linkedin",""),("nb_author_github",""),
            ("nb_author_kaggle",""),("nb_author_facebook",""),
        ]:
            if k not in st.session_state: st.session_state[k]=v

        # ── Gather session data for JS ──
        has_results = st.session_state.results is not None
        _ds = str(st.session_state.dataset_name or "No dataset loaded")
        _rows = len(df)
        _target = str(st.session_state.get("target_col") or df.columns[0])
        _bm = "—"
        _score = "—"
        _metric = "—"
        _ptype_detected = st.session_state.problem_type or "classification"

        if has_results:
            res_nb = st.session_state.results
            mc_nb = "Model" if "Model" in res_nb.columns else res_nb.columns[0]
            nr_nb = res_nb.select_dtypes(include=[np.number]).columns.tolist()
            _bm = str(res_nb.iloc[0][mc_nb])
            _metric = nr_nb[0] if nr_nb else "Score"
            _score = f"{res_nb.iloc[0][_metric]:.4f}" if nr_nb else "—"

        # ── Streamlit state → JS init values ──
        _nb_step = st.session_state.nb_step
        _nb_eda  = st.session_state.nb_eda
        _nb_pre  = st.session_state.nb_pre
        _nb_train= st.session_state.nb_train
        _nb_export= st.session_state.nb_export
        _nb_ptype= st.session_state.nb_ptype

        # ── Build init JS for states ──
        eda_init = str(_nb_eda).lower().replace("true","true").replace("false","false").replace("'",'"')
        pre_init = str(_nb_pre).lower().replace("true","true").replace("false","false").replace("'",'"')
        train_init= str(_nb_train).lower().replace("true","true").replace("false","false").replace("'",'"')
        export_init=str(_nb_export).lower().replace("true","true").replace("false","false").replace("'",'"')

        # ── Render the Pro Notebook Builder HTML ──
        nb_builder_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:#080b12;color:#e2e8f0;font-family:'Space Grotesk',sans-serif;overflow-x:hidden;}}
:root{{--bg0:#080b12;--bg1:#0d1117;--bg2:#111827;--bg3:#1a2234;--border:#1e3a5f;--border2:#2d4f7c;--green:#22d3a0;--blue:#38bdf8;--purple:#a78bfa;--amber:#fbbf24;--red:#f87171;--text1:#f0f6ff;--text2:#94a3b8;--text3:#475569;}}
.shell{{display:grid;grid-template-columns:240px 1fr;min-height:600px;max-height:860px;overflow:hidden;}}
/* RAIL */
.rail{{background:var(--bg1);border-right:1px solid var(--border);display:flex;flex-direction:column;min-width:0;overflow:hidden;}}
.rail-logo{{padding:.85rem 1.1rem .75rem;border-bottom:1px solid var(--border);}}
.rail-logo .lt{{font-size:.82rem;font-weight:700;background:linear-gradient(135deg,var(--green),var(--blue));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.rail-logo .ls{{font-size:.58rem;color:var(--text3);margin-top:1px;text-transform:uppercase;letter-spacing:.07em;}}
.rail-steps{{padding:.65rem 0;flex:1;overflow-y:auto;}}
.sr{{display:flex;align-items:flex-start;gap:.5rem;padding:.45rem 1rem;cursor:pointer;transition:background .15s;position:relative;min-width:0;}}
.sr:hover{{background:rgba(255,255,255,.03);}}
.sr.active{{background:linear-gradient(90deg,rgba(34,211,160,.08),transparent);}}
.sr.active::before{{content:'';position:absolute;left:0;top:0;bottom:0;width:2.5px;background:linear-gradient(180deg,var(--green),var(--blue));}}
.sd{{width:24px;height:24px;min-width:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:700;flex-shrink:0;border:1.5px solid var(--border2);background:var(--bg2);color:var(--text3);transition:all .25s;}}
.sr.done .sd{{background:rgba(34,211,160,.15);border-color:var(--green);color:var(--green);}}
.sr.active .sd{{background:rgba(56,189,248,.15);border-color:var(--blue);color:var(--blue);box-shadow:0 0 8px rgba(56,189,248,.3);}}
.sc{{width:1.5px;height:8px;background:var(--border);margin:1px 0 1px 23px;}}
.sc.done{{background:var(--green);}}
.si{{flex:1;min-width:0;overflow:hidden;}}
.stitle{{font-size:.72rem;font-weight:600;color:var(--text2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.sr.active .stitle{{color:var(--text1);}}
.sdesc{{font-size:.6rem;color:var(--text3);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.rail-footer{{padding:.85rem 1.1rem;border-top:1px solid var(--border);}}
.prog-label{{font-size:.58rem;color:var(--text3);text-transform:uppercase;letter-spacing:.07em;margin-bottom:.35rem;display:flex;justify-content:space-between;}}
.prog-bar{{height:3px;background:var(--bg3);border-radius:2px;overflow:hidden;}}
.prog-fill{{height:100%;background:linear-gradient(90deg,var(--green),var(--blue));border-radius:2px;transition:width .5s cubic-bezier(.4,0,.2,1);}}
/* MAIN */
.main{{background:var(--bg0);display:flex;flex-direction:column;min-width:0;overflow:hidden;}}
.topbar{{background:var(--bg1);border-bottom:1px solid var(--border);padding:.55rem 1.25rem;display:flex;align-items:center;justify-content:space-between;gap:.5rem;flex-shrink:0;}}
.tb-left{{min-width:0;flex:1;}}
.tb-bc{{font-size:.6rem;color:var(--text3);}}
.tb-sn{{font-size:.78rem;font-weight:700;color:var(--text1);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.tb-right{{display:flex;align-items:center;gap:.4rem;flex-shrink:0;}}
.chip{{display:inline-flex;align-items:center;padding:.15rem .5rem;border-radius:4px;font-size:.6rem;font-weight:600;border:1px solid;white-space:nowrap;}}
.chip-g{{background:rgba(34,211,160,.1);color:var(--green);border-color:rgba(34,211,160,.25);}}
.chip-b{{background:rgba(56,189,248,.1);color:var(--blue);border-color:rgba(56,189,248,.25);}}
.chip-p{{background:rgba(167,139,250,.1);color:var(--purple);border-color:rgba(167,139,250,.25);}}
.content{{flex:1;overflow-y:auto;padding:1.1rem 1.25rem;}}
/* PANELS */
.panel{{display:none;animation:fadeIn .25s ease;}}
.panel.active{{display:block;}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:none}}}}
.ph{{margin-bottom:1.1rem;}}
.ph h2{{font-size:1rem;font-weight:700;color:var(--text1);margin-bottom:.2rem;}}
.ph p{{font-size:.72rem;color:var(--text2);line-height:1.55;}}
/* CARDS */
.card{{background:var(--bg1);border:1px solid var(--border);border-radius:11px;padding:1rem;margin-bottom:.75rem;}}
.ct{{font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--text3);margin-bottom:.75rem;display:flex;align-items:center;gap:.4rem;}}
.ct::after{{content:'';flex:1;height:1px;background:var(--border);}}
/* RADIO PILLS — GRID, NO OVERFLOW */
.rg{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:.55rem;}}
.rp{{display:flex;align-items:center;gap:.5rem;padding:.65rem .8rem;border-radius:9px;border:1.5px solid var(--border2);cursor:pointer;transition:all .18s;min-width:0;overflow:hidden;width:100%;}}
.rp:hover{{border-color:var(--blue);background:rgba(56,189,248,.05);}}
.rp.sel{{border-color:var(--green);background:rgba(34,211,160,.07);}}
.rp input{{display:none;}}
.pdot{{width:14px;height:14px;min-width:14px;border-radius:50%;border:2px solid var(--border2);display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .18s;position:relative;}}
.rp.sel .pdot{{border-color:var(--green);}}
.rp.sel .pdot::after{{content:'';width:6px;height:6px;border-radius:50%;background:var(--green);position:absolute;}}
.picon{{font-size:1rem;flex-shrink:0;}}
.ptext{{min-width:0;overflow:hidden;}}
.plabel{{font-size:.75rem;font-weight:600;color:var(--text2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.rp.sel .plabel{{color:var(--text1);}}
.psub{{font-size:.6rem;color:var(--text3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
/* STAT ROW — NO OVERFLOW */
.stat-row{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.5rem;margin-bottom:.75rem;}}
.sm{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:.55rem .65rem;min-width:0;overflow:hidden;}}
.sm-l{{font-size:.55rem;color:var(--text3);text-transform:uppercase;letter-spacing:.07em;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.sm-v{{font-size:.85rem;font-weight:700;margin-top:.15rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.sm-g{{color:var(--green);}}.sm-b{{color:var(--blue);}}.sm-p{{color:var(--purple);}}.sm-a{{color:var(--amber);}}
/* TOGGLE GRID */
.tg{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:.45rem;}}
.ti{{display:flex;align-items:center;justify-content:space-between;padding:.55rem .75rem;background:var(--bg2);border:1px solid var(--border);border-radius:8px;cursor:pointer;transition:all .18s;gap:.35rem;min-width:0;overflow:hidden;}}
.ti:hover{{border-color:var(--border2);}}
.ti.on{{border-color:rgba(34,211,160,.3);background:rgba(34,211,160,.04);}}
.tl{{display:flex;align-items:center;gap:.4rem;min-width:0;overflow:hidden;flex:1;}}
.tico{{font-size:.8rem;flex-shrink:0;}}
.tlbl{{font-size:.68rem;font-weight:500;color:var(--text2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.ti.on .tlbl{{color:var(--text1);}}
.tbadge{{font-size:.52rem;padding:.08rem .3rem;border-radius:3px;background:rgba(167,139,250,.15);color:var(--purple);border:1px solid rgba(167,139,250,.25);font-weight:700;flex-shrink:0;}}
.tsw{{width:28px;height:15px;min-width:28px;background:var(--bg3);border-radius:7px;position:relative;transition:background .18s;flex-shrink:0;}}
.tsw::after{{content:'';position:absolute;left:2px;top:2px;width:11px;height:11px;border-radius:50%;background:var(--text3);transition:all .18s;}}
.ti.on .tsw{{background:var(--green);}}
.ti.on .tsw::after{{left:15px;background:#fff;}}
/* SECTION DIV */
.sdiv{{display:flex;align-items:center;gap:.5rem;margin:.8rem 0 .55rem;}}
.sdiv-l{{font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;white-space:nowrap;}}
.sdiv-g{{color:var(--green);}}.sdiv-b{{color:var(--blue);}}
.sdiv::before,.sdiv::after{{content:'';flex:1;height:1px;background:var(--border);}}
/* INPUTS */
.ig{{margin-bottom:.7rem;}}
.il{{font-size:.65rem;font-weight:600;color:var(--text2);margin-bottom:.3rem;display:block;}}
.ifield{{width:100%;background:var(--bg2);border:1px solid var(--border2);border-radius:6px;padding:.5rem .7rem;color:var(--text1);font-size:.73rem;font-family:'Space Grotesk',sans-serif;outline:none;transition:border-color .18s,box-shadow .18s;}}
.ifield:focus{{border-color:var(--blue);box-shadow:0 0 0 2px rgba(56,189,248,.1);}}
.ifield::placeholder{{color:var(--text3);}}
.igrid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:.6rem;}}
/* PREVIEW BAR */
.pbar{{background:var(--bg1);border:1px solid var(--border);border-radius:9px;padding:.6rem .85rem;margin-bottom:.75rem;display:flex;align-items:center;gap:.6rem;min-width:0;overflow:hidden;}}
.pb-label{{font-size:.58rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text3);flex-shrink:0;}}
.pb-chips{{display:flex;flex-wrap:wrap;gap:.25rem;flex:1;min-width:0;overflow:hidden;max-height:46px;}}
.pb-chip{{padding:.12rem .4rem;border-radius:4px;font-size:.58rem;font-weight:600;background:rgba(34,211,160,.1);color:var(--green);border:1px solid rgba(34,211,160,.2);white-space:nowrap;}}
.pb-count{{font-size:.68rem;font-weight:700;color:var(--text1);white-space:nowrap;flex-shrink:0;}}
.pb-count span{{color:var(--green);}}
/* NOTEBOOK PREVIEW */
.nbp{{background:var(--bg1);border:1px solid var(--border);border-radius:11px;overflow:hidden;margin-bottom:.75rem;}}
.nbp-bar{{background:var(--bg2);padding:.4rem .75rem;display:flex;align-items:center;gap:.35rem;border-bottom:1px solid var(--border);}}
.nbdot{{width:8px;height:8px;border-radius:50%;}}
.dr{{background:#f87171;}}.dy{{background:#fbbf24;}}.dg{{background:#4ade80;}}
.nbfn{{font-size:.62rem;color:var(--text3);margin-left:.35rem;font-family:'JetBrains Mono',monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.nb-cells{{padding:.65rem;display:flex;flex-direction:column;gap:.35rem;max-height:280px;overflow-y:auto;}}
.nbc{{border-radius:6px;font-size:.62rem;font-family:'JetBrains Mono',monospace;position:relative;overflow:hidden;}}
.nbc-md{{background:rgba(167,139,250,.06);border:1px solid rgba(167,139,250,.2);padding:.4rem .65rem;color:var(--purple);}}
.nbc-code{{background:rgba(56,189,248,.05);border:1px solid rgba(56,189,248,.15);padding:.4rem .65rem .4rem 1.8rem;color:var(--blue);}}
.nbc-code::before{{content:'[ ]';position:absolute;left:.45rem;top:.38rem;color:var(--text3);font-size:.55rem;}}
.nbc-inner{{color:var(--text2);}}
.nbc-active{{border-color:var(--green);animation:cp 2s infinite;}}
@keyframes cp{{0%,100%{{box-shadow:0 0 0 0 rgba(34,211,160,0)}}50%{{box-shadow:0 0 0 3px rgba(34,211,160,.1)}}}}
/* EXPORT SUMMARY */
.esw{{background:var(--bg1);border:1px solid var(--border);border-radius:11px;padding:1rem;margin-bottom:.75rem;}}
.esr{{display:flex;justify-content:space-between;align-items:center;padding:.38rem 0;border-bottom:1px solid var(--border);gap:.4rem;}}
.esr:last-child{{border-bottom:none;}}
.esk{{font-size:.68rem;color:var(--text3);flex-shrink:0;}}
.esv{{font-size:.68rem;font-weight:600;color:var(--text1);font-family:'JetBrains Mono',monospace;text-align:right;min-width:0;overflow:hidden;text-overflow:ellipsis;}}
.esv.g{{color:var(--green);}}
/* AUTHOR PREVIEW */
.ap{{background:var(--bg2);border:1px solid var(--border);border-radius:9px;padding:.85rem;display:flex;align-items:center;gap:.75rem;margin-bottom:.75rem;min-width:0;overflow:hidden;}}
.av{{width:38px;height:38px;min-width:38px;border-radius:50%;background:linear-gradient(135deg,var(--green),var(--blue));display:flex;align-items:center;justify-content:center;font-size:1rem;font-weight:700;color:#080b12;flex-shrink:0;}}
.ai{{min-width:0;flex:1;overflow:hidden;}}
.an{{font-size:.78rem;font-weight:700;color:var(--text1);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.atitle{{font-size:.62rem;color:var(--text3);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.sc-wrap{{display:flex;gap:.3rem;flex-wrap:wrap;margin-top:.4rem;}}
.soc{{padding:.12rem .4rem;border-radius:3px;font-size:.58rem;font-weight:600;border:1px solid;white-space:nowrap;}}
/* BUTTONS */
.btn-row{{display:flex;align-items:center;gap:.55rem;margin-top:1rem;flex-wrap:wrap;}}
.btn-back{{padding:.5rem 1rem;border-radius:7px;border:1px solid var(--border2);background:transparent;color:var(--text2);font-size:.73rem;font-weight:600;cursor:pointer;font-family:'Space Grotesk',sans-serif;transition:all .18s;white-space:nowrap;}}
.btn-back:hover{{border-color:var(--text2);color:var(--text1);}}
.btn-next{{padding:.5rem 1.2rem;border-radius:7px;border:none;background:linear-gradient(135deg,#22d3a0,#38bdf8);color:#080b12;font-size:.75rem;font-weight:700;cursor:pointer;font-family:'Space Grotesk',sans-serif;transition:all .18s;box-shadow:0 3px 14px rgba(34,211,160,.3);white-space:nowrap;}}
.btn-next:hover{{transform:translateY(-1px);box-shadow:0 6px 20px rgba(34,211,160,.45);}}
.btn-dl{{padding:.65rem 1.25rem;border-radius:9px;border:none;background:linear-gradient(135deg,#667eea,#764ba2,#f093fb);color:#fff;font-size:.8rem;font-weight:700;cursor:pointer;font-family:'Space Grotesk',sans-serif;transition:all .18s;box-shadow:0 5px 18px rgba(102,126,234,.4);letter-spacing:.02em;width:100%;margin-top:.25rem;}}
.btn-dl:hover{{transform:translateY(-2px);box-shadow:0 10px 26px rgba(102,126,234,.55);filter:brightness(1.07);}}
.note-box{{background:rgba(56,189,248,.04);border:1px solid rgba(56,189,248,.2);border-radius:9px;padding:.75rem 1rem;font-size:.7rem;color:var(--text2);line-height:1.6;margin-bottom:.75rem;}}
code{{color:var(--blue);background:rgba(56,189,248,.1);padding:.08rem .3rem;border-radius:3px;font-family:'JetBrains Mono',monospace;}}
::-webkit-scrollbar{{width:3px;height:3px;}}::-webkit-scrollbar-track{{background:var(--bg0);}}::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:2px;}}
</style>
</head>
<body>
<div class="shell">

<!-- RAIL -->
<div class="rail">
  <div class="rail-logo">
    <div class="lt">⚡ DataForge ML Studio</div>
    <div class="ls">Notebook Builder Pro</div>
  </div>
  <div class="rail-steps" id="railSteps"></div>
  <div class="rail-footer">
    <div class="prog-label"><span>Progress</span><span id="progPct">0%</span></div>
    <div class="prog-bar"><div class="prog-fill" id="progFill" style="width:0%"></div></div>
  </div>
</div>

<!-- MAIN -->
<div class="main">
  <div class="topbar">
    <div class="tb-left">
      <div class="tb-bc">Notebook Builder</div>
      <div class="tb-sn" id="topStep">Step 1 of 6 — Problem Type</div>
    </div>
    <div class="tb-right">
      <span class="chip chip-g" id="chipCells">~42 cells</span>
      <span class="chip chip-b" id="chipPtype">Classification</span>
      <span class="chip chip-p">📓 .ipynb</span>
    </div>
  </div>

  <div class="content">

    <!-- P1: PROBLEM TYPE -->
    <div class="panel active" id="panel1">
      <div class="ph"><h2>🎯 Select Problem Type</h2><p>Choose the ML task for your notebook. DataForge auto-detects from your target column — override here if needed.</p></div>
      <div class="card">
        <div class="ct">Task</div>
        <div class="rg">
          <label class="rp sel" id="pill-clf" onclick="selPtype('classification')">
            <input type="radio" name="pt" value="classification" checked>
            <div class="pdot"></div><div class="picon">🎯</div>
            <div class="ptext"><div class="plabel">Classification</div><div class="psub">Predict categories/labels</div></div>
          </label>
          <label class="rp" id="pill-reg" onclick="selPtype('regression')">
            <input type="radio" name="pt" value="regression">
            <div class="pdot"></div><div class="picon">📈</div>
            <div class="ptext"><div class="plabel">Regression</div><div class="psub">Predict continuous values</div></div>
          </label>
        </div>
      </div>
      <div class="card">
        <div class="ct">Session Snapshot</div>
        <div class="stat-row">
          <div class="sm"><div class="sm-l">Dataset</div><div class="sm-v sm-b">{_ds[:16]}</div></div>
          <div class="sm"><div class="sm-l">Rows</div><div class="sm-v sm-g">{_rows:,}</div></div>
          <div class="sm"><div class="sm-l">Target</div><div class="sm-v sm-p">{_target[:10]}</div></div>
          <div class="sm"><div class="sm-l">Best Model</div><div class="sm-v sm-a">{_bm[:10]}</div></div>
        </div>
      </div>
      <div class="btn-row"><button class="btn-next" onclick="go(2)">Continue →</button></div>
    </div>

    <!-- P2: EDA -->
    <div class="panel" id="panel2">
      <div class="ph"><h2>🧬 EDA — Charts & Analyses</h2><p>Select which exploratory analysis charts to include in your generated notebook.</p></div>
      <div class="pbar">
        <div class="pb-label">Included</div>
        <div class="pb-chips" id="edaChips"></div>
        <div class="pb-count"><span id="edaCount">10</span> selected</div>
      </div>
      <div class="card">
        <div class="ct">Basic Charts</div>
        <div class="tg">
          <div class="ti on" data-key="distributions" onclick="togEDA(this)"><div class="tl"><div class="tico">📊</div><div class="tlbl">Distributions histogram</div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="correlation" onclick="togEDA(this)"><div class="tl"><div class="tico">🔥</div><div class="tlbl">Correlation heatmap</div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="missing" onclick="togEDA(this)"><div class="tl"><div class="tico">❓</div><div class="tlbl">Missing values chart</div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="target_dist" onclick="togEDA(this)"><div class="tl"><div class="tico">🎯</div><div class="tlbl">Target distribution</div></div><div class="tsw"></div></div>
          <div class="ti" data-key="boxplots" onclick="togEDA(this)"><div class="tl"><div class="tico">📦</div><div class="tlbl">Box plots</div></div><div class="tsw"></div></div>
          <div class="ti" data-key="scatter_matrix" onclick="togEDA(this)"><div class="tl"><div class="tico">🔵</div><div class="tlbl">Scatter matrix</div></div><div class="tsw"></div></div>
          <div class="ti" data-key="cat_bars" onclick="togEDA(this)"><div class="tl"><div class="tico">📋</div><div class="tlbl">Category bar charts</div></div><div class="tsw"></div></div>
          <div class="ti" data-key="outlier_plot" onclick="togEDA(this)"><div class="tl"><div class="tico">⚠️</div><div class="tlbl">Outlier detection plot</div></div><div class="tsw"></div></div>
        </div>
      </div>
      <div class="sdiv"><div class="sdiv-l sdiv-g">🔬 Advanced EDA</div></div>
      <div class="card">
        <div class="ct">Advanced Analyses <span style="font-size:.55rem;color:var(--purple);padding:.08rem .35rem;border-radius:3px;background:rgba(167,139,250,.1);border:1px solid rgba(167,139,250,.2);margin-left:.3rem;">NEW</span></div>
        <div class="tg">
          <div class="ti on" data-key="violin_plots" onclick="togEDA(this)"><div class="tl"><div class="tico">🎻</div><div class="tlbl">Violin plots <span class="tbadge">NEW</span></div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="skewness_kurtosis" onclick="togEDA(this)"><div class="tl"><div class="tico">📐</div><div class="tlbl">Skewness & Kurtosis <span class="tbadge">NEW</span></div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="numeric_summary_styled" onclick="togEDA(this)"><div class="tl"><div class="tico">📊</div><div class="tlbl">Enhanced num summary <span class="tbadge">NEW</span></div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="iqr_analysis" onclick="togEDA(this)"><div class="tl"><div class="tico">📏</div><div class="tlbl">IQR outlier table <span class="tbadge">NEW</span></div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="feature_importance_eda" onclick="togEDA(this)"><div class="tl"><div class="tico">⭐</div><div class="tlbl">Quick RF importance <span class="tbadge">NEW</span></div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="categorical_summary" onclick="togEDA(this)"><div class="tl"><div class="tico">🔤</div><div class="tlbl">Categorical summary <span class="tbadge">NEW</span></div></div><div class="tsw"></div></div>
          <div class="ti" data-key="value_counts_table" onclick="togEDA(this)"><div class="tl"><div class="tico">📋</div><div class="tlbl">Full value counts</div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="outlier_table" onclick="togEDA(this)"><div class="tl"><div class="tico">🔎</div><div class="tlbl">Outlier sample rows <span class="tbadge">NEW</span></div></div><div class="tsw"></div></div>
        </div>
      </div>
      <div class="btn-row"><button class="btn-back" onclick="go(1)">← Back</button><button class="btn-next" onclick="go(3)">Continue →</button></div>
    </div>

    <!-- P3: PREPROCESSING -->
    <div class="panel" id="panel3">
      <div class="ph"><h2>🧹 Preprocessing Steps</h2><p>Choose which preprocessing operations to include as code cells in the notebook.</p></div>
      <div class="card">
        <div class="ct">Operations</div>
        <div class="tg">
          <div class="ti on" data-key="drop_dups" onclick="togPRE(this)"><div class="tl"><div class="tico">🗑️</div><div class="tlbl">Drop duplicate rows</div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="handle_missing" onclick="togPRE(this)"><div class="tl"><div class="tico">🩹</div><div class="tlbl">Handle missing values</div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="normalize" onclick="togPRE(this)"><div class="tl"><div class="tico">⚖️</div><div class="tlbl">Normalize features</div></div><div class="tsw"></div></div>
          <div class="ti" data-key="remove_outliers" onclick="togPRE(this)"><div class="tl"><div class="tico">🚫</div><div class="tlbl">Remove outliers (reg)</div></div><div class="tsw"></div></div>
        </div>
      </div>
      <div class="note-box">💡 PyCaret handles most preprocessing automatically during <code>setup()</code>. These add explicit cells before the AutoML pipeline — useful for Kaggle notebooks where step-by-step clarity matters.</div>
      <div class="btn-row"><button class="btn-back" onclick="go(2)">← Back</button><button class="btn-next" onclick="go(4)">Continue →</button></div>
    </div>

    <!-- P4: TRAINING CHARTS -->
    <div class="panel" id="panel4">
      <div class="ph"><h2>📊 Training & Results Charts</h2><p>Select which result visualizations to include in the model comparison section.</p></div>
      <div class="card">
        <div class="ct">Visualizations</div>
        <div class="tg">
          <div class="ti on" data-key="model_table" onclick="togTR(this)"><div class="tl"><div class="tico">📋</div><div class="tlbl">Models comparison table</div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="bar_chart" onclick="togTR(this)"><div class="tl"><div class="tico">📊</div><div class="tlbl">Top models bar chart</div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="radar_chart" onclick="togTR(this)"><div class="tl"><div class="tico">🕸️</div><div class="tlbl">Metrics radar chart</div></div><div class="tsw"></div></div>
          <div class="ti" data-key="feature_importance" onclick="togTR(this)"><div class="tl"><div class="tico">⭐</div><div class="tlbl">Feature importance plot</div></div><div class="tsw"></div></div>
        </div>
      </div>
      <div class="nbp">
        <div class="nbp-bar"><div class="nbdot dr"></div><div class="nbdot dy"></div><div class="nbdot dg"></div><div class="nbfn">dataforge_{_ds[:12].replace(' ','_')}_{_bm[:10].replace(' ','_')}.ipynb</div></div>
        <div class="nb-cells">
          <div class="nbc nbc-md">📦 Section 1 — Install & Import Libraries</div>
          <div class="nbc nbc-code"><div class="nbc-inner">import pandas as pd, numpy as np ...</div></div>
          <div class="nbc nbc-md">🧬 Section 4 — Exploratory Data Analysis</div>
          <div class="nbc nbc-code"><div class="nbc-inner">fig = px.histogram(df, x=col) ...</div></div>
          <div class="nbc nbc-md">⚙️ Section 6 — AutoML Training</div>
          <div class="nbc nbc-code nbc-active"><div class="nbc-inner">clf_compare(verbose=True) ✦ generating...</div></div>
          <div class="nbc nbc-md">🏆 Section 7 — Results & Visualizations</div>
          <div class="nbc nbc-code"><div class="nbc-inner">px.bar(top6, x=metric_name ...) ...</div></div>
        </div>
      </div>
      <div class="btn-row"><button class="btn-back" onclick="go(3)">← Back</button><button class="btn-next" onclick="go(5)">Continue →</button></div>
    </div>

    <!-- P5: AUTHOR -->
    <div class="panel" id="panel5">
      <div class="ph"><h2>🧑‍💻 About the Author</h2><p>These details appear in a styled section at the end of the notebook. All fields are optional.</p></div>
      <div class="ap">
        <div class="av" id="authAv">?</div>
        <div class="ai">
          <div class="an" id="authNm">Your Name</div>
          <div class="atitle" id="authTt">Your Title</div>
          <div class="sc-wrap" id="authSc"></div>
        </div>
      </div>
      <div class="card">
        <div class="ct">Personal Info</div>
        <div class="igrid">
          <div class="ig"><label class="il">👤 Your Name</label><input class="ifield" id="inp-name" placeholder="e.g. Muhammad Shayan" oninput="updAuth()"></div>
          <div class="ig"><label class="il">💼 Your Title</label><input class="ifield" id="inp-title" placeholder="e.g. AI Engineer" oninput="updAuth()"></div>
          <div class="ig"><label class="il">💡 Favorite Quote</label><input class="ifield" id="inp-quote" placeholder="Inspire your readers..."></div>
          <div class="ig"><label class="il">📧 Email</label><input class="ifield" id="inp-email" placeholder="you@email.com" oninput="updSoc()"></div>
        </div>
      </div>
      <div class="sdiv"><div class="sdiv-l sdiv-b">Social Links</div></div>
      <div class="card">
        <div class="ct">Profiles</div>
        <div class="igrid">
          <div class="ig"><label class="il">🔗 LinkedIn URL</label><input class="ifield" id="inp-linkedin" placeholder="https://linkedin.com/in/..." oninput="updSoc()"></div>
          <div class="ig"><label class="il">💻 GitHub URL</label><input class="ifield" id="inp-github" placeholder="https://github.com/..." oninput="updSoc()"></div>
          <div class="ig"><label class="il">🧠 Kaggle URL</label><input class="ifield" id="inp-kaggle" placeholder="https://kaggle.com/..." oninput="updSoc()"></div>
          <div class="ig"><label class="il">📘 Facebook URL</label><input class="ifield" id="inp-facebook" placeholder="https://facebook.com/..." oninput="updSoc()"></div>
        </div>
      </div>
      <div class="btn-row"><button class="btn-back" onclick="go(4)">← Back</button><button class="btn-next" onclick="go(6)">Continue →</button></div>
    </div>

    <!-- P6: EXPORT -->
    <div class="panel" id="panel6">
      <div class="ph"><h2>💾 Export Options</h2><p>Final configuration. Review your selections and download.</p></div>
      <div class="esw">
        <div class="ct" style="margin-bottom:.6rem;">📋 Notebook Summary</div>
        <div class="esr"><span class="esk">Dataset</span><span class="esv">{_ds[:24]}</span></div>
        <div class="esr"><span class="esk">Problem Type</span><span class="esv" id="esPtype">Classification</span></div>
        <div class="esr"><span class="esk">Best Model</span><span class="esv g">{_bm}</span></div>
        <div class="esr"><span class="esk">Score ({_metric})</span><span class="esv g">{_score}</span></div>
        <div class="esr"><span class="esk">EDA Analyses</span><span class="esv g" id="esEda">—</span></div>
        <div class="esr"><span class="esk">Estimated Cells</span><span class="esv g" id="esCells">—</span></div>
        <div class="esr"><span class="esk">Author</span><span class="esv" id="esAuthor">Anonymous</span></div>
      </div>
      <div class="card">
        <div class="ct">Include in Export</div>
        <div class="tg">
          <div class="ti on" data-key="save_model" onclick="togEX(this)"><div class="tl"><div class="tico">💾</div><div class="tlbl">Save model (.pkl)</div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="load_predict" onclick="togEX(this)"><div class="tl"><div class="tico">🔮</div><div class="tlbl">Load & predict example</div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="summary_table" onclick="togEX(this)"><div class="tl"><div class="tico">📋</div><div class="tlbl">Session summary table</div></div><div class="tsw"></div></div>
          <div class="ti on" data-key="author_section" onclick="togEX(this)"><div class="tl"><div class="tico">🧑‍💻</div><div class="tlbl">About the author</div></div><div class="tsw"></div></div>
        </div>
      </div>
      {"<button class='btn-dl' id='dlBtn' onclick='trigDL()'>📓 Download Notebook (.ipynb)</button>" if has_results else "<div class='note-box' style='text-align:center;'>⚠️ Please train a model in the <strong>⚙️ Train Model</strong> tab first, then come back to download your notebook!</div>"}
      <div class="btn-row" style="margin-top:.65rem;">
        <button class="btn-back" onclick="go(5)">← Back</button>
        {"" if has_results else "<div style='font-size:.62rem;color:var(--text3);'>Train → come back → download 🚀</div>"}
      </div>
    </div>

  </div>
</div>
</div>

<script>
const STEPS=[
  {{n:1,title:"Problem Type",desc:"Classification or Regression",icon:"🎯"}},
  {{n:2,title:"EDA Charts",desc:"Choose analyses",icon:"🧬"}},
  {{n:3,title:"Preprocessing",desc:"Data cleaning steps",icon:"🧹"}},
  {{n:4,title:"Training Charts",desc:"Result visualizations",icon:"📊"}},
  {{n:5,title:"Author Info",desc:"Optional — for Kaggle",icon:"🧑‍💻"}},
  {{n:6,title:"Export",desc:"Download notebook",icon:"💾"}},
];

let cur={_nb_step};
let ptv="{_nb_ptype}";
let edaS={{}},preS={{}},trS={{}},exS={{}};

// Init states from Python
const edaInit={eda_init};
const preInit={pre_init};
const trInit={train_init};
const exInit={export_init};
Object.keys(edaInit).forEach(k=>{{edaS[k]=edaInit[k];}});
Object.keys(preInit).forEach(k=>{{preS[k]=preInit[k];}});
Object.keys(trInit).forEach(k=>{{trS[k]=trInit[k];}});
Object.keys(exInit).forEach(k=>{{exS[k]=exInit[k];}});

// Sync DOM toggles with state
function syncToggleDom(){{
  document.querySelectorAll('[data-key]').forEach(el=>{{
    const k=el.dataset.key;
    let state=null;
    if(edaInit.hasOwnProperty(k)) state=edaS[k];
    else if(preInit.hasOwnProperty(k)) state=preS[k];
    else if(trInit.hasOwnProperty(k)) state=trS[k];
    else if(exInit.hasOwnProperty(k)) state=exS[k];
    if(state!==null){{ if(state) el.classList.add('on'); else el.classList.remove('on'); }}
  }});
}}

function buildRail(){{
  const r=document.getElementById('railSteps'); let h='';
  STEPS.forEach((s,i)=>{{
    const done=s.n<cur,active=s.n===cur;
    const cls=done?'done':active?'active':''; const ic=done?'✓':s.n;
    h+=`<div class="sr ${{cls}}" onclick="go(${{s.n}})"><div class="sd">${{ic}}</div><div class="si"><div class="stitle">${{s.icon}} ${{s.title}}</div><div class="sdesc">${{s.desc}}</div></div></div>`;
    if(i<STEPS.length-1) h+=`<div class="sc ${{done?'done':''}}"></div>`;
  }});
  r.innerHTML=h;
}}

function go(n){{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.getElementById('panel'+n).classList.add('active');
  cur=n; buildRail(); updTopbar(); updProg(); updExport();
}}

function updTopbar(){{
  const s=STEPS[cur-1];
  document.getElementById('topStep').textContent=`Step ${{cur}} of 6 — ${{s.title}}`;
  document.getElementById('chipPtype').textContent=ptv==='classification'?'Classification':'Regression';
  const cnt=Object.values(edaS).filter(Boolean).length;
  const total=6+cnt*2+Object.values(preS).filter(Boolean).length+Object.values(trS).filter(Boolean).length+Object.values(exS).filter(Boolean).length+12;
  document.getElementById('chipCells').textContent='~'+total+' cells';
}}

function updProg(){{
  const pct=Math.round((cur-1)/5*100);
  document.getElementById('progFill').style.width=pct+'%';
  document.getElementById('progPct').textContent=pct+'%';
}}

function selPtype(v){{
  ptv=v;
  document.getElementById('pill-clf').classList.toggle('sel',v==='classification');
  document.getElementById('pill-reg').classList.toggle('sel',v==='regression');
  updTopbar();
}}

const EDA_LBL={{distributions:"📊 Dist",correlation:"🔥 Corr",missing:"❓ Missing",target_dist:"🎯 Target",boxplots:"📦 Box",scatter_matrix:"🔵 Scatter",cat_bars:"📋 Cat",outlier_plot:"⚠️ Outlier",violin_plots:"🎻 Violin",skewness_kurtosis:"📐 Skew",numeric_summary_styled:"📊 NumSum",iqr_analysis:"📏 IQR",feature_importance_eda:"⭐ RF Imp",categorical_summary:"🔤 CatSum",value_counts_table:"📋 VC",outlier_table:"🔎 OutRows"}};

function updEdaPreview(){{
  const on=Object.keys(edaS).filter(k=>edaS[k]);
  document.getElementById('edaCount').textContent=on.length;
  document.getElementById('edaChips').innerHTML=on.map(k=>`<div class="pb-chip">${{EDA_LBL[k]||k}}</div>`).join('');
}}

function togEDA(el){{el.classList.toggle('on');edaS[el.dataset.key]=el.classList.contains('on');updEdaPreview();updTopbar();}}
function togPRE(el){{el.classList.toggle('on');preS[el.dataset.key]=el.classList.contains('on');}}
function togTR(el){{el.classList.toggle('on');trS[el.dataset.key]=el.classList.contains('on');}}
function togEX(el){{el.classList.toggle('on');exS[el.dataset.key]=el.classList.contains('on');updExport();}}

function updAuth(){{
  const nm=document.getElementById('inp-name').value||'Your Name';
  const tt=document.getElementById('inp-title').value||'Your Title';
  document.getElementById('authNm').textContent=nm;
  document.getElementById('authTt').textContent=tt;
  const ini=nm.split(' ').map(w=>w[0]||'').join('').slice(0,2).toUpperCase()||'?';
  document.getElementById('authAv').textContent=ini;
}}

function updSoc(){{
  const socs=[
    {{id:'inp-email',l:'📧',c:'rgba(248,113,113,.1)',bc:'rgba(248,113,113,.3)',tc:'#f87171'}},
    {{id:'inp-linkedin',l:'🔗',c:'rgba(56,189,248,.1)',bc:'rgba(56,189,248,.3)',tc:'#38bdf8'}},
    {{id:'inp-github',l:'💻',c:'rgba(148,163,184,.1)',bc:'rgba(148,163,184,.3)',tc:'#94a3b8'}},
    {{id:'inp-kaggle',l:'🧠',c:'rgba(34,211,160,.1)',bc:'rgba(34,211,160,.3)',tc:'#22d3a0'}},
    {{id:'inp-facebook',l:'📘',c:'rgba(96,165,250,.1)',bc:'rgba(96,165,250,.3)',tc:'#60a5fa'}},
  ];
  document.getElementById('authSc').innerHTML=socs.filter(s=>document.getElementById(s.id)?.value.trim()).map(s=>`<div class="soc" style="background:${{s.c}};border-color:${{s.bc}};color:${{s.tc}}">${{s.l}}</div>`).join('');
}}

function updExport(){{
  const n=Object.values(edaS).filter(Boolean).length;
  document.getElementById('esEda').textContent=n+' analyses';
  const total=6+n*2+Object.values(preS).filter(Boolean).length+Object.values(trS).filter(Boolean).length+Object.values(exS).filter(Boolean).length+12;
  document.getElementById('esCells').textContent='~'+total;
  document.getElementById('esPtype').textContent=ptv==='classification'?'Classification':'Regression';
  const nm=document.getElementById('inp-name')?.value;
  document.getElementById('esAuthor').textContent=nm||'Anonymous';
}}

function trigDL(){{
  const btn=document.getElementById('dlBtn');
  if(!btn) return;
  btn.textContent='⏳ Generating...';btn.style.opacity='.7';btn.disabled=true;
  setTimeout(()=>{{
    btn.textContent='✅ Use the Streamlit download button in the Results tab!';
    btn.style.background='linear-gradient(135deg,#22d3a0,#38bdf8)';btn.style.color='#080b12';btn.style.opacity='1';
    setTimeout(()=>{{btn.textContent='📓 Download Notebook (.ipynb)';btn.style.background='linear-gradient(135deg,#667eea,#764ba2,#f093fb)';btn.style.color='#fff';btn.disabled=false;}},3500);
  }},1200);
}}

// Init
syncToggleDom();buildRail();updTopbar();updProg();updEdaPreview();updExport();
if(ptv==='regression'){{selPtype('regression');}}
</script>
</body>
</html>
"""

        st.components.v1.html(nb_builder_html, height=920, scrolling=False)

        # ── Download button below the HTML component ──
        if has_results:
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""<div style="background:{"rgba(74,222,128,0.06)" if T=="dark" else "rgba(124,58,237,0.06)"};border:1px solid {"rgba(74,222,128,0.25)" if T=="dark" else "rgba(124,58,237,0.25)"};border-radius:14px;padding:1.25rem 1.5rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
              <div><div style="font-size:.85rem;font-weight:800;color:{ACCENT1}">✅ Model Trained — Notebook Ready!</div>
              <div style="font-size:.75rem;color:{TEXT2};margin-top:.2rem">Configure above → click download below</div></div>
            </div>""", unsafe_allow_html=True)

            res_nb = st.session_state.results
            mc_nb  = "Model" if "Model" in res_nb.columns else res_nb.columns[0]
            nr_nb  = res_nb.select_dtypes(include=[np.number]).columns.tolist()
            bn_nb  = res_nb.iloc[0][mc_nb]
            mn_nb  = nr_nb[0] if nr_nb else "Score"
            ts_nb  = res_nb.iloc[0][mn_nb] if nr_nb else 0

            try:
                target_for_nb = st.session_state.get("target_col") or df.columns[0]
                nb_bytes = generate_notebook(
                    df=df, target_col=target_for_nb,
                    problem_type=st.session_state.nb_ptype,
                    results_df=res_nb, best_model_name=bn_nb,
                    top_score=ts_nb, metric_name=mn_nb,
                    dataset_name=str(st.session_state.dataset_name or "dataset"),
                    training_time=st.session_state.training_time or 0,
                    fold=st.session_state.cv_fold or 5,
                    normalize=st.session_state.nb_pre.get("normalize", True),
                    train_size=0.8,
                    eda_selections=st.session_state.nb_eda,
                    pre_selections=st.session_state.nb_pre,
                    train_selections=st.session_state.nb_train,
                    export_selections=st.session_state.nb_export,
                    author_name=st.session_state.get("nb_author_name",""),
                    author_title=st.session_state.get("nb_author_title",""),
                    author_quote=st.session_state.get("nb_author_quote",""),
                    author_email=st.session_state.get("nb_author_email",""),
                    author_linkedin=st.session_state.get("nb_author_linkedin",""),
                    author_github=st.session_state.get("nb_author_github",""),
                    author_kaggle=st.session_state.get("nb_author_kaggle",""),
                    author_facebook=st.session_state.get("nb_author_facebook",""),
                )
                safe_nm=str(st.session_state.dataset_name or "dataset").replace(" ","_").replace(".","_")
                nb_filename=f"dataforge_{safe_nm}_{bn_nb.replace(' ','_')}.ipynb"
                st.download_button(
                    label="📓 Download Notebook (.ipynb)",
                    data=nb_bytes,
                    file_name=nb_filename,
                    mime="application/x-ipynb+json",
                    key="nb_builder_download_main"
                )
            except Exception as nb_err:
                st.error(f"❌ Notebook generation failed: {nb_err}")

            # ── AI Insights (Claude API) ──
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""<div style="background:{'rgba(167,139,250,0.07)' if T=='dark' else 'rgba(124,58,237,0.06)'};border:1px solid {'rgba(167,139,250,0.3)' if T=='dark' else 'rgba(124,58,237,0.25)'};border-radius:16px;padding:1.25rem 1.5rem;margin-bottom:1rem;">
              <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.5rem">
                <span style="font-size:1.5rem">🤖</span>
                <div>
                  <div style="font-size:.9rem;font-weight:800;color:{'#c084fc' if T=='dark' else '#7c3aed'}">AI Insights — Powered by CodGenZ</div>
                  <div style="font-size:.72rem;color:{TEXT3}">Get deep AI analysis of your trained model and dataset</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Build context for Claude
            res_for_ai = st.session_state.results
            mc_ai = "Model" if "Model" in res_for_ai.columns else res_for_ai.columns[0]
            nr_ai = res_for_ai.select_dtypes(include=[np.number]).columns.tolist()
            bn_ai = res_for_ai.iloc[0][mc_ai]
            mn_ai = nr_ai[0] if nr_ai else "Score"
            ts_ai = float(res_for_ai.iloc[0][mn_ai]) if nr_ai else 0.0
            tc_ai = st.session_state.get("target_col") or df.columns[0]
            pt_ai = st.session_state.problem_type or "classification"
            ds_ai = str(st.session_state.dataset_name or "dataset")
            tt_ai = st.session_state.training_time or 0
            folds_ai = st.session_state.cv_fold or 5
            null_pct_ai = round(df.isnull().sum().sum() / df.size * 100, 2)
            dup_ai = int(df.duplicated().sum())
            num_cols_ai = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols_ai = df.select_dtypes(include=["object","category"]).columns.tolist()

            # Top 5 models summary
            try:
                top5_str = res_for_ai[[mc_ai] + nr_ai[:4]].head(5).to_string(index=False)
            except:
                top5_str = str(res_for_ai.head(5))

            ai_context = f"""Dataset: {ds_ai}
Rows: {len(df):,} | Columns: {len(df.columns)}
Numerical features: {len(num_cols_ai)} | Categorical features: {len(cat_cols_ai)}
Missing values: {null_pct_ai}% | Duplicate rows: {dup_ai}
Target column: {tc_ai}
Problem type: {pt_ai}
Best model: {bn_ai}
Best score ({mn_ai}): {ts_ai:.4f}
CV Folds: {folds_ai}
Training time: {tt_ai:.1f}s

Top 5 Models:
{top5_str}"""

            ai_col1, ai_col2 = st.columns([3,1])
            with ai_col1:
                ai_insight_type = st.selectbox(
                    "Select Insight Type",
                    ["🔍 Full Model Analysis", "📊 Data Quality Report", "🏆 Model Comparison Deep Dive",
                     "💡 Feature Engineering Tips", "🚀 Deployment Recommendations", "📈 How to Improve Score"],
                    key="ai_insight_type_select",
                    label_visibility="collapsed"
                )
            with ai_col2:
                ai_clicked = st.button("✨ Generate AI Insights", key="ai_insights_btn")

            if "ai_insight_result" not in st.session_state:
                st.session_state.ai_insight_result = None
            if "ai_insight_type_last" not in st.session_state:
                st.session_state.ai_insight_type_last = None

            if ai_clicked:
                insight_prompts = {
                    "🔍 Full Model Analysis": f"""You are an expert ML engineer analyzing results from DataForge ML Studio.

{ai_context}

Provide a comprehensive analysis including:
1. **Model Performance Assessment** — Is {ts_ai:.4f} {mn_ai} good for this type of problem? What does it mean practically?
2. **Data Quality Insights** — Issues and concerns from the dataset stats
3. **Why {bn_ai} Won** — Explain why this algorithm likely outperformed others
4. **Overfitting Risk** — Assess the risk based on dataset size and score
5. **Top 3 Actionable Recommendations** — Specific steps to improve further
6. **Kaggle Competitiveness** — How would this model perform in a real competition?

Use emojis, be specific, and give practical advice. Format with clear headings.""",

                    "📊 Data Quality Report": f"""You are a data scientist reviewing dataset quality.

{ai_context}

Provide a detailed data quality report:
1. **Overall Health Score** (0-100) with reasoning
2. **Missing Values Analysis** — Impact and recommended imputation strategies
3. **Feature Mix Assessment** — {len(num_cols_ai)} numerical + {len(cat_cols_ai)} categorical — is this balanced?
4. **Potential Data Leakage Risks** — Common pitfalls for {pt_ai} problems
5. **Recommended Preprocessing Steps** — Specific to this dataset profile
6. **Data Collection Improvements** — What additional data would help most?

Be specific and actionable. Use emojis.""",

                    "🏆 Model Comparison Deep Dive": f"""You are an AutoML expert explaining model selection.

{ai_context}

Analyze the model comparison results:
1. **Why {bn_ai} is Best** — Technical explanation of its strengths for this problem
2. **Runner-up Analysis** — What the 2nd best model does differently
3. **Tree vs Linear vs Ensemble** — Which family won and why it suits this data
4. **Hyperparameter Tuning Potential** — How much improvement is possible with tuning {bn_ai}?
5. **Ensemble Opportunity** — Would stacking these models help?
6. **Training Efficiency** — Was {tt_ai:.1f}s training time reasonable?

Be technical but understandable.""",

                    "💡 Feature Engineering Tips": f"""You are a feature engineering specialist.

{ai_context}

Suggest feature engineering improvements:
1. **Top 3 New Features to Create** — Specific to this dataset ({ds_ai}, target: {tc_ai})
2. **Interaction Features** — Which numerical pairs to multiply/divide
3. **Categorical Encoding Improvements** — Beyond basic label encoding
4. **Date/Time Features** (if applicable) — Temporal patterns to extract
5. **Feature Selection** — Which features might be hurting the model
6. **Target Encoding** — When and how to use it for {pt_ai}
7. **Expected Score Improvement** — Realistic estimate after these changes

Give very specific, code-hintable suggestions.""",

                    "🚀 Deployment Recommendations": f"""You are an MLOps engineer reviewing this model for production.

{ai_context}

Provide deployment guidance:
1. **Production Readiness Score** (0-100) — Is {ts_ai:.4f} {mn_ai} good enough for production?
2. **Serving Strategy** — REST API vs batch vs real-time for this use case
3. **Monitoring Requirements** — What metrics to track in production
4. **Retraining Schedule** — How often should this model be retrained?
5. **Infrastructure Needs** — Estimated compute for {bn_ai} serving
6. **A/B Testing Plan** — How to safely roll out this model
7. **Risk Assessment** — What could go wrong and mitigation strategies

Be practical and specific.""",

                    "📈 How to Improve Score": f"""You are an ML competition expert.

{ai_context}

Current score: {ts_ai:.4f} {mn_ai} with {bn_ai}

Give a step-by-step improvement roadmap:
1. **Quick Wins (1-2 hours)** — Immediate changes for +score
2. **Hyperparameter Tuning** — Specific parameters to tune for {bn_ai}
3. **Feature Engineering Priority** — Top 3 features to add
4. **Data Augmentation** — If applicable for {pt_ai}
5. **Advanced Ensemble Methods** — Stacking, blending strategy
6. **Expected Score Range** — Realistic target after all improvements
7. **Effort vs Impact Matrix** — Which improvements give best ROI

Be very specific with numbers and parameters."""
                }

                prompt = insight_prompts.get(ai_insight_type, insight_prompts["🔍 Full Model Analysis"])

                _api_key = st.secrets.get("GROQ_API_KEY", "")
                if not _api_key:
                    st.error("❌ GROQ_API_KEY not found in secrets! Add it in App Settings → Secrets")
                    st.stop()
                with st.spinner("🤖 CodeXpert AI is analyzing your ML results..."):
                    try:
                        import requests as _req
                        resp = _req.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {_api_key}"
                            },
                            json={
                                "model": "llama-3.3-70b-versatile",
                                "max_tokens": 1500,
                                "messages": [{"role": "user", "content": prompt}]
                            },
                            timeout=60
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            insight_text = data["choices"][0]["message"]["content"]
                            st.session_state.ai_insight_result = insight_text
                            st.session_state.ai_insight_type_last = ai_insight_type
                        else:
                            st.error(f"❌ API Error {resp.status_code}: {resp.text[:200]}")
                    except Exception as ai_err:
                        st.error(f"❌ Could not connect to Claude API: {ai_err}")

            # Display AI Insights result
            if st.session_state.ai_insight_result:
                ai_bg = "rgba(167,139,250,0.06)" if T=="dark" else "rgba(124,58,237,0.04)"
                ai_border = "rgba(167,139,250,0.25)" if T=="dark" else "rgba(124,58,237,0.20)"
                ai_type_lbl = st.session_state.get("ai_insight_type_last","AI Insights")
                insight_txt = st.session_state.ai_insight_result

                # Render with forced white text in dark card
                st.markdown(f"""
                <div style="background:{ai_bg};border:1px solid {ai_border};border-radius:14px;padding:1.5rem 1.75rem;margin-top:.75rem;">
                  <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{'#a78bfa' if T=='dark' else '#7c3aed'};margin-bottom:1rem;">{ai_type_lbl}</div>
                  <div style="color:{'#f0f6ff' if T=='dark' else '#111827'};font-size:.88rem;line-height:1.8;white-space:pre-wrap;">{insight_txt}</div>
                </div>""", unsafe_allow_html=True)

                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

                # Buttons row — Regenerate + Download
                regen_col1, regen_col2, regen_col3 = st.columns([1,1,3])
                with regen_col1:
                    if st.button("🔄 Regenerate", key="ai_regen_btn"):
                        st.session_state.ai_insight_result = None
                        st.rerun()
                with regen_col2:
                    # Download as .txt
                    dl_txt = ai_type_lbl + "\n" + "="*60 + "\n\n" + insight_txt
                    st.download_button(
                        label="📥 Download",
                        data=dl_txt.encode("utf-8"),
                        file_name=f"ai_insights_{ai_type_lbl[:20].replace(' ','_').replace('/','')}.txt",
                        mime="text/plain",
                        key="ai_download_btn"
                    )

            # ── Star Rating ──
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            _au = str(st.session_state.get("nb_author_name","Anonymous"))
            st.components.v1.html(f"""
<!DOCTYPE html><html><head>
<script src="https://cdn.jsdelivr.net/npm/@emailjs/browser@4/dist/email.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&display=swap');
*{{box-sizing:border-box;margin:0;padding:0;font-family:'Space Grotesk',sans-serif;}}
body{{background:transparent;}}
.rc{{background:{"#0d0d0d" if T=="dark" else "#ffffff"};border:1px solid {"#222" if T=="dark" else "#e5e7eb"};border-radius:16px;padding:1.5rem;position:relative;overflow:hidden;}}
.rc::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#667eea,#f093fb,#4ade80,#60a5fa);background-size:300% 100%;animation:sh 3s linear infinite;}}
@keyframes sh{{0%{{background-position:0% 0%}}100%{{background-position:300% 0%}}}}
.rc-title{{text-align:center;font-size:1rem;font-weight:800;color:{"#f9fafb" if T=="dark" else "#111827"};margin-bottom:.2rem;}}
.rc-sub{{text-align:center;font-size:.73rem;color:{"#6b7280" if T=="dark" else "#9ca3af"};margin-bottom:1rem;}}
.stars-wrap{{display:flex;justify-content:center;gap:8px;margin-bottom:.75rem;}}
.star{{font-size:2.2rem;cursor:pointer;color:#d1d5db;transition:transform .2s,color .15s,filter .2s;user-select:none;}}
.star:hover,.star.hov{{color:#fbbf24;transform:scale(1.25);filter:drop-shadow(0 0 8px #fbbf2488);}}
.star.sel{{color:#fbbf24;transform:scale(1.1);}}
@keyframes sp{{0%{{transform:scale(1)}}40%{{transform:scale(1.5) rotate(-8deg)}}70%{{transform:scale(.95) rotate(4deg)}}100%{{transform:scale(1.1)}}}}
.star.pop{{animation:sp .4s cubic-bezier(.34,1.56,.64,1) both;}}
.rl{{text-align:center;font-size:.85rem;font-weight:700;min-height:1.2rem;margin-bottom:.7rem;color:#4ade80;}}
textarea{{width:100%;background:{"#141414" if T=="dark" else "#f9fafb"};border:1px solid {"#2a2a2a" if T=="dark" else "#e5e7eb"};border-radius:9px;color:{"#f9fafb" if T=="dark" else "#111827"};padding:.6rem .85rem;font-size:.78rem;resize:vertical;min-height:70px;outline:none;font-family:'Space Grotesk',sans-serif;transition:border-color .2s;margin-bottom:.75rem;}}
textarea:focus{{border-color:#667eea;}}
textarea::placeholder{{color:{"#4b5563" if T=="dark" else "#9ca3af"};}}
.sb{{display:block;width:100%;padding:.75rem;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;font-size:.85rem;font-weight:700;border:none;border-radius:9px;cursor:pointer;box-shadow:0 4px 16px rgba(102,126,234,.4);transition:transform .15s,box-shadow .15s;}}
.sb:hover{{transform:translateY(-2px);box-shadow:0 8px 24px rgba(102,126,234,.55);}}
.sb:disabled{{background:#374151;color:#6b7280;cursor:not-allowed;box-shadow:none;transform:none;}}
.ty{{text-align:center;padding:1.25rem .75rem;animation:fu .5s ease both;}}
@keyframes fu{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:none}}}}
.ty-stars{{font-size:2rem;margin-bottom:.5rem;letter-spacing:3px;}}
.ty-msg{{font-size:.95rem;font-weight:800;color:#4ade80;margin-bottom:.2rem;}}
.ty-sub{{font-size:.7rem;color:{"#6b7280" if T=="dark" else "#9ca3af"};}}
.rb{{margin-top:.85rem;background:none;border:1px solid {"#333" if T=="dark" else "#e5e7eb"};color:{"#9ca3af" if T=="dark" else "#6b7280"};padding:.35rem .9rem;border-radius:7px;font-size:.7rem;cursor:pointer;transition:border-color .2s,color .2s;}}
.rb:hover{{border-color:#667eea;color:#667eea;}}
.status{{font-size:.7rem;text-align:center;margin-top:.5rem;min-height:1rem;}}
.ok{{color:#4ade80;}}.err{{color:#f87171;}}.sending{{color:#60a5fa;}}
</style></head><body>
<div class="rc">
  <div id="rForm">
    <div class="rc-title">⭐ Rate Your Experience</div>
    <div class="rc-sub">How was DataForge ML Studio?</div>
    <div class="stars-wrap" id="sw">
      <span class="star" data-v="1">★</span><span class="star" data-v="2">★</span><span class="star" data-v="3">★</span><span class="star" data-v="4">★</span><span class="star" data-v="5">★</span>
    </div>
    <div class="rl" id="rl"></div>
    <textarea id="ftxt" placeholder="Share your thoughts (optional)..." style="display:none"></textarea>
    <button class="sb" id="sb" style="display:none" disabled>🚀 Submit Rating</button>
    <div class="status" id="stMsg"></div>
  </div>
  <div class="ty" id="ty" style="display:none">
    <div class="ty-stars" id="tyS"></div>
    <div class="ty-msg" id="tyM"></div>
    <div class="ty-sub" id="tySub"></div>
    <br><button class="rb" onclick="reset()">✏️ Update Rating</button>
  </div>
</div>
<script>
emailjs.init("KPju9potPVtR0LXSX");
const LBL=["","😞 Poor","😕 Fair","😊 Good","😄 Great","🤩 Excellent!"];
const CLR=["","#f87171","#fb923c","#fbbf24","#4ade80","#4ade80"];
let sel=0;
const stars=document.querySelectorAll('.star');
const rl=document.getElementById('rl'),ft=document.getElementById('ftxt'),sb=document.getElementById('sb'),stMsg=document.getElementById('stMsg'),rForm=document.getElementById('rForm'),ty=document.getElementById('ty');
stars.forEach(s=>{{
  s.addEventListener('mouseenter',()=>{{const v=+s.dataset.v;stars.forEach(x=>x.classList.toggle('hov',+x.dataset.v<=v));rl.textContent=LBL[v];rl.style.color=CLR[v];}});
  s.addEventListener('mouseleave',()=>{{stars.forEach(x=>x.classList.remove('hov'));rl.textContent=sel?LBL[sel]:'';rl.style.color=sel?CLR[sel]:'';}});
  s.addEventListener('click',()=>{{sel=+s.dataset.v;stars.forEach((x,i)=>{{x.classList.remove('sel','pop','hov');if(i<sel){{x.classList.add('sel');setTimeout(()=>x.classList.add('pop'),i*55);}}}});rl.textContent=LBL[sel];rl.style.color=CLR[sel];ft.style.display='block';sb.style.display='block';sb.disabled=false;}});
}});
sb.addEventListener('click',()=>{{
  if(!sel)return;sb.disabled=true;sb.textContent='⏳ Sending...';stMsg.textContent='Sending...';stMsg.className='status sending';
  const se='⭐'.repeat(sel)+'☆'.repeat(5-sel),fb=ft.value.trim()||'No feedback.',now=new Date().toLocaleString('en-PK',{{timeZone:'Asia/Karachi'}});
  emailjs.send('service_7gw3npx','template_qlievo7',{{to_email:'shayan.corner@gmail.com',from_name:'{_au}',rating:sel+' / 5  '+se,label:LBL[sel],feedback:fb,dataset:'{_ds}',best_model:'{_bm}',score:'{_score}',problem:'{_ptype_detected}',submitted_at:now}}).then(()=>{{stMsg.textContent='';showTY(fb);}},()=>{{sb.disabled=false;sb.textContent='🚀 Submit Rating';stMsg.textContent='⚠️ Could not send — retrying in 2s';stMsg.className='status err';setTimeout(()=>showTY(fb),2000);}});
}});
function showTY(fb){{
  rForm.style.display='none';ty.style.display='block';
  const se='⭐'.repeat(sel)+'☆'.repeat(5-sel);
  document.getElementById('tyS').textContent=se;
  const msgs=["","🙏 We'll improve!","🙏 We'll improve!","😊 Thanks!","😄 Thanks a lot!","🎉 You made our day!"];
  document.getElementById('tyM').textContent=msgs[sel];
  document.getElementById('tySub').textContent='Rating: '+sel+'/5 — '+LBL[sel];
}}
function reset(){{sel=0;stars.forEach(x=>x.classList.remove('sel','pop','hov'));rl.textContent='';ft.value='';ft.style.display='none';sb.style.display='none';stMsg.textContent='';rForm.style.display='block';ty.style.display='none';}}
</script>
</body></html>""", height=340, scrolling=False)

else:
    # ── WELCOME SCREEN ──
    st.markdown(f"""
    <div style="text-align:center;min-height:40vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem 2rem" class="slide-up">
      <div style="font-size:3.8rem;font-weight:900;letter-spacing:-.04em;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 1rem;line-height:1.05">Drop Your Data.<br>We Do the Rest.</div>
      <p style="font-size:1.1rem;color:{TEXT2};max-width:560px;line-height:1.7;margin:0 0 2rem">DataForge ML Studio — zero-code AutoML. Upload a CSV, pick a target, hit train. Get a production model in minutes. <b style="color:{ACCENT1}">Everything is 100% free.</b></p>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
    f1,f2,f3,f4=st.columns(4)
    feats=[("🧬","Smart EDA","Correlation heatmaps, distribution explorer, outlier detection, advanced stats."),("⚡","AutoML Engine","15+ algorithms compared with k-fold cross-validation. Best model wins automatically."),("🎯","Smart Detection","Auto-detects regression vs classification. Warns about ID columns. Quick data cleaning."),("🏆","Rich Results","Trophy banner, radar + scatter + bar charts, metric breakdown, model export (.pkl).")]
    for col,(icon,title,desc) in zip([f1,f2,f3,f4],feats):
        with col:
            st.markdown(f"""<div class="feature-card slide-up"><div class="fc-icon">{icon}</div><h3>{title}</h3><p>{desc}</p></div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""<div style="background:{"rgba(74,222,128,0.06)" if T=="dark" else "rgba(124,58,237,0.06)"};border:2px solid {"rgba(74,222,128,0.25)" if T=="dark" else "rgba(124,58,237,0.25)"};border-radius:20px;padding:2rem;text-align:center;margin-bottom:2rem">
      <div style="font-size:1.4rem;font-weight:900;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:1rem">🎁 Everything Free. No Login. No Limits.</div>
      <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:.75rem">
        {"".join(f'<span class="insight-chip" style="border-color:{ACCENT1};color:{ACCENT1}">{f}</span>' for f in ["✅ XGBoost","✅ LightGBM","✅ CatBoost","✅ 10-fold CV","✅ Model Export (.pkl)","✅ Notebook Export (.ipynb)","✅ 13+ Algorithms","✅ No Sign-up Required"])}
      </div>
    </div>
    <div style="text-align:center;color:{TEXT3};font-size:.82rem;padding-bottom:1.5rem">👈 Upload a CSV/Excel or load a sample dataset from the sidebar to get started</div>""", unsafe_allow_html=True)
