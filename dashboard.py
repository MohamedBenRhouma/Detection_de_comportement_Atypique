# -*- coding: utf-8 -*-
# =============================================================================
#  BANKING FRAUD DETECTION DASHBOARD
#  Built with: Streamlit + Plotly
#  Run with:   streamlit run dashboard.py
# =============================================================================

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st
import pandas as pd
pd.set_option("styler.render.max_elements", 1000000)
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Banking Fraud Detection | UIB",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — premium dark-blue banking theme ──────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background-color: #0d1117; }
  section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

  /* ── Header banner ── */
  .header-banner {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0,120,255,0.15);
  }
  .header-banner h1 { color: #58a6ff; margin: 0; font-size: 2rem; font-weight: 700; }
  .header-banner p  { color: #8b949e; margin: 6px 0 0; font-size: 0.95rem; }

  /* ── KPI cards ── */
  .kpi-card {
    background: linear-gradient(135deg, #161b22, #1c2330);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,120,255,0.2); }
  .kpi-value { font-size: 2.1rem; font-weight: 700; margin: 6px 0; }
  .kpi-label { color: #8b949e; font-size: 0.82rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.8px; }
  .kpi-delta { font-size: 0.78rem; margin-top: 4px; }

  .kpi-blue   .kpi-value { color: #58a6ff; }
  .kpi-green  .kpi-value { color: #3fb950; }
  .kpi-red    .kpi-value { color: #f85149; }
  .kpi-orange .kpi-value { color: #d29922; }
  .kpi-purple .kpi-value { color: #bc8cff; }

  /* ── Section titles ── */
  .section-title {
    color: #c9d1d9;
    font-size: 1.1rem;
    font-weight: 600;
    border-left: 4px solid #58a6ff;
    padding-left: 12px;
    margin: 24px 0 16px;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"]  { background: #161b22; border-radius: 10px; gap: 6px; }
  .stTabs [data-baseweb="tab"]       { color: #8b949e; border-radius: 8px; padding: 8px 20px; font-weight: 500; }
  .stTabs [aria-selected="true"]     { background: #1f6feb !important; color: #ffffff !important; }

  /* ── Sidebar ── */
  .sidebar-title { color: #58a6ff; font-size: 0.85rem; font-weight: 600;
                   text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }

  /* ── Alert boxes ── */
  .alert-red    { background:#2d1117; border:1px solid #f85149; border-radius:10px; padding:14px; color:#f85149; }
  .alert-green  { background:#0d2318; border:1px solid #3fb950; border-radius:10px; padding:14px; color:#3fb950; }
  .alert-orange { background:#2d1f00; border:1px solid #d29922; border-radius:10px; padding:14px; color:#d29922; }
  .alert-blue   { background:#0d1f3c; border:1px solid #58a6ff; border-radius:10px; padding:14px; color:#58a6ff; }

  /* ── Scrollable table ─- */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* ── Plotly chart background ── */
  .js-plotly-plot .plotly .bg { fill: #161b22 !important; }

  /* ── Download button ── */
  .stDownloadButton button {
    background: #1f6feb; color: white; border: none;
    border-radius: 8px; padding: 8px 20px; font-weight: 600;
    transition: background 0.2s;
  }
  .stDownloadButton button:hover { background: #388bfd; }

  /* ── Footer ── */
  .footer { color: #484f58; font-size: 0.75rem; text-align: center; margin-top: 40px;
            border-top: 1px solid #21262d; padding-top: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme helper ────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Inter"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="#1c2330", bordercolor="#30363d", borderwidth=1),
    colorway=["#58a6ff","#3fb950","#f85149","#d29922","#bc8cff","#79c0ff"],
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)


# =============================================================================
# DATA LOADING
# =============================================================================

RESULTS_FILE = "fraud_detection_results.xlsx"
DATA_FILE    = "Data.xlsm"

@st.cache_data(show_spinner="Loading data ...")
def load_results():
    """Load the enriched results file produced by main.py."""
    if not os.path.exists(RESULTS_FILE):
        return None, "run_main"
    try:
        df = pd.read_excel(RESULTS_FILE, sheet_name="All Transactions", engine="openpyxl")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        
        # --- Generate Explanations for Anomalies ---
        if "anomaly_IF" in df.columns:
            reasons = pd.Series([""] * len(df), index=df.index)
            normal_mask = df["anomaly_IF"] == 1
            reasons[normal_mask] = "Normal transaction"
            
            anom_mask = df["anomaly_IF"] == -1
            
            if "CV EN TND" in df.columns:
                mnt_95 = df["CV EN TND"].quantile(0.95)
                mnt_05 = df["CV EN TND"].quantile(0.05)
                reasons = reasons.mask(anom_mask & (df["CV EN TND"] > mnt_95), reasons + " • Unusually high amount")
                reasons = reasons.mask(anom_mask & (df["CV EN TND"] < mnt_05), reasons + " • Unusually small amount")
                
            if "high_risk_country" in df.columns:
                reasons = reasons.mask(anom_mask & (df["high_risk_country"] == 1), reasons + " • High-risk origin country")
                
            
            if "D_EXEC" in df.columns and "Date" in df.columns:
                delay = (pd.to_datetime(df["D_EXEC"], errors='coerce') - pd.to_datetime(df["Date"], errors='coerce')).dt.days
                delay_95 = delay.quantile(0.95)
                reasons = reasons.mask(anom_mask & (delay > delay_95), reasons + " • Long processing delay")

            # --- New engineered feature explanations ---
            if "txn_count_sender_7d" in df.columns:
                vel_95 = df["txn_count_sender_7d"].quantile(0.95)
                reasons = reasons.mask(anom_mask & (df["txn_count_sender_7d"] > vel_95), reasons + " • High sender velocity (7d burst)")

            if "amount_vs_sender_avg" in df.columns:
                reasons = reasons.mask(anom_mask & (df["amount_vs_sender_avg"] > 5), reasons + " • Amount 5x+ sender average")

            if "country_mismatch_risk" in df.columns:
                reasons = reasons.mask(anom_mask & (df["country_mismatch_risk"] == 1), reasons + " • Country mismatch (BIC vs declared)")

            if "is_round_amount" in df.columns:
                reasons = reasons.mask(anom_mask & (df["is_round_amount"] == 1), reasons + " • Perfectly round amount")

            if "rare_currency_for_country" in df.columns:
                reasons = reasons.mask(anom_mask & (df["rare_currency_for_country"] == 1), reasons + " • Unusual currency for country")

            reasons = reasons.str.replace(r"^ • ", "", regex=True)
            empty_anom_mask = anom_mask & (reasons == "")
            if "anomaly_score_IF" in df.columns:
                highly_iso = df["anomaly_score_IF"] < -0.10
                reasons = reasons.mask(empty_anom_mask & highly_iso, "Highly isolated pattern")
                reasons = reasons.mask(empty_anom_mask & ~highly_iso, "Unusual multivariate profile")
            else:
                reasons = reasons.mask(empty_anom_mask, "Unusual multivariate profile")
                
            df["Explanation"] = reasons
            
        return df, "ok"
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner="Loading raw data ...")
def load_raw():
    """Load the raw DETAIL sheet directly from Data.xlsm."""
    try:
        df = pd.read_excel(DATA_FILE, sheet_name="DETAIL", engine="openpyxl")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    except Exception:
        return None


df, status = load_results()


# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="header-banner">
  <h1>🏦 Banking Fraud Detection Dashboard</h1>
  <p>Isolation Forest + Mahalanobis Distance &nbsp;|&nbsp; UIB Tunisia &nbsp;|&nbsp;
     SWIFT Wire Transfer Analysis &nbsp;|&nbsp; PFE — Final Year Project</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# GUARD: results file must exist
# =============================================================================

if status == "run_main":
    st.markdown("""
    <div class="alert-orange">
      <b>⚠ Results file not found.</b><br>
      Please run <code>python -X utf8 main.py</code> first to generate
      <code>results_with_anomalies.xlsx</code>, then refresh this page.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if df is None:
    st.error(f"Could not load results: {status}")
    st.stop()


# =============================================================================
# SIDEBAR — FILTERS
# =============================================================================

with st.sidebar:
    st.markdown('<div class="sidebar-title">🔍 Filters</div>', unsafe_allow_html=True)

    # Date range
    if df["Date"].notna().any():
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            df = df[(df["Date"].dt.date >= date_range[0]) &
                    (df["Date"].dt.date <= date_range[1])]

    st.markdown("---")

    # Country
    if "Nom_pays" in df.columns:
        all_countries = sorted(df["Nom_pays"].dropna().unique().tolist())
        sel_countries = st.multiselect("Origin Country", all_countries, default=[])
        if sel_countries:
            df = df[df["Nom_pays"].isin(sel_countries)]

    # Currency
    if "DEV" in df.columns:
        all_dev = sorted(df["DEV"].dropna().unique().tolist())
        sel_dev = st.multiselect("Currency (DEV)", all_dev, default=[])
        if sel_dev:
            df = df[df["DEV"].isin(sel_dev)]

    # FRAIS
    if "FRAIS" in df.columns:
        all_frais = sorted(df["FRAIS"].dropna().unique().tolist())
        sel_frais = st.multiselect("Fee Type (FRAIS)", all_frais, default=[])
        if sel_frais:
            df = df[df["FRAIS"].isin(sel_frais)]

    st.markdown("---")

    # Anomaly filter
    anomaly_filter = st.radio(
        "Show transactions",
        ["All", "Anomalies Only", "Normal Only", "High Confidence Fraud"],
        index=0
    )

    # Amount range
    if "CV EN TND" in df.columns and df["CV EN TND"].notna().any():
        amt_min = float(df["CV EN TND"].min())
        amt_max = float(df["CV EN TND"].max())
        if amt_min == amt_max:
            amt_range = (amt_min, amt_max)
        else:
            amt_range = st.slider("CV EN TND range", amt_min, amt_max,
                                  (amt_min, amt_max), format="%.0f")
        df = df[(df["CV EN TND"] >= amt_range[0]) & (df["CV EN TND"] <= amt_range[1])]

    st.markdown("---")
    st.markdown('<div class="sidebar-title">📁 Export</div>', unsafe_allow_html=True)

    if st.button("♻️ Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# Apply anomaly filter
df_display = df.copy()
if anomaly_filter == "Anomalies Only":
    df_display = df_display[df_display["anomaly_IF"] == -1]
elif anomaly_filter == "Normal Only":
    df_display = df_display[df_display["anomaly_IF"] == 1]
elif anomaly_filter == "High Confidence Fraud":
    df_display = df_display[df_display["high_confidence_fraud"] == 1]


# =============================================================================
# KPI CARDS
# =============================================================================

n_total    = len(df)
n_anom     = int((df["anomaly_IF"] == -1).sum())  if "anomaly_IF"           in df.columns else 0
n_high     = int(df["high_confidence_fraud"].sum()) if "high_confidence_fraud" in df.columns else 0
total_tnd  = df["CV EN TND"].sum()                 if "CV EN TND"            in df.columns else 0
anom_tnd   = df[df["anomaly_IF"] == -1]["CV EN TND"].sum() if "anomaly_IF"  in df.columns else 0
# rpa_is_problem removed — not a valid feature
method_agr = round(np.mean(
    df["anomaly_IF"] == df["anomaly_Distance"]) * 100, 1) if "anomaly_Distance" in df.columns else None

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="kpi-card kpi-blue">
      <div class="kpi-label">Total Transactions</div>
      <div class="kpi-value">{n_total:,}</div>
      <div class="kpi-delta" style="color:#58a6ff">SWIFT wire transfers</div>
    </div>""", unsafe_allow_html=True)

with col2:
    pct = n_anom / n_total * 100 if n_total else 0
    st.markdown(f"""
    <div class="kpi-card kpi-red">
      <div class="kpi-label">IF Anomalies</div>
      <div class="kpi-value">{n_anom:,}</div>
      <div class="kpi-delta" style="color:#f85149">{pct:.2f}% of total</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card kpi-orange">
      <div class="kpi-label">High Confidence Fraud</div>
      <div class="kpi-value">{n_high:,}</div>
      <div class="kpi-delta" style="color:#d29922">Both methods agree</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card kpi-green">
      <div class="kpi-label">Total Volume (TND)</div>
      <div class="kpi-value">{total_tnd/1e6:.1f}M</div>
      <div class="kpi-delta" style="color:#3fb950">Suspicious: {anom_tnd/1e6:.1f}M TND</div>
    </div>""", unsafe_allow_html=True)

with col5:
    if method_agr is not None:
        st.markdown(f"""
        <div class="kpi-card kpi-purple">
          <div class="kpi-label">Method Agreement</div>
          <div class="kpi-value">{method_agr}%</div>
          <div class="kpi-delta" style="color:#bc8cff">IF vs Mahalanobis</div>
        </div>""", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔴 Anomaly Analysis",
    "🗺️  Geography",
    "🤖 Model Evaluation & Insights",
    "🧪 Simulator",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Volume Over Time</div>', unsafe_allow_html=True)

    if "Date" in df.columns and df["Date"].notna().any():
        # Daily transaction volume
        daily = (df.groupby(df["Date"].dt.date)
                   .agg(count=("CHRONO","count"), volume=("CV EN TND","sum"))
                   .reset_index())
        daily.columns = ["Date","Count","Volume_TND"]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=daily["Date"], y=daily["Count"],
                             name="# Transactions", marker_color="#1f6feb", opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Volume_TND"],
                                 name="Volume (TND)", line=dict(color="#f0883e", width=2),
                                 mode="lines"), secondary_y=True)
        fig.update_layout(title="Daily Transaction Count & Volume", **PLOTLY_LAYOUT,
                          height=350, barmode="overlay")
        fig.update_yaxes(title_text="# Transactions", secondary_y=False,
                         gridcolor="#21262d", linecolor="#30363d")
        fig.update_yaxes(title_text="Volume TND", secondary_y=True,
                         gridcolor="#21262d", linecolor="#30363d")
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Currency (DEV) Distribution</div>', unsafe_allow_html=True)
        if "DEV" in df.columns:
            dev_counts = df["DEV"].value_counts().reset_index()
            dev_counts.columns = ["Currency", "Count"]
            fig = px.pie(dev_counts, values="Count", names="Currency",
                         color_discrete_sequence=["#58a6ff","#3fb950","#f85149","#d29922","#bc8cff","#79c0ff","#56d364"])
            fig.update_layout(**PLOTLY_LAYOUT, height=320,
                              title="Transactions by Currency")
            fig.update_traces(textfont_color="#c9d1d9", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Fee Type (FRAIS) Breakdown</div>', unsafe_allow_html=True)
        if "FRAIS" in df.columns:
            frais_anom = (df.groupby("FRAIS")["anomaly_IF"]
                           .apply(lambda x: (x == -1).sum())
                           .reset_index())
            frais_norm = (df.groupby("FRAIS")["anomaly_IF"]
                           .apply(lambda x: (x == 1).sum())
                           .reset_index())
            frais_anom.columns = ["FRAIS","Anomaly"]
            frais_norm.columns = ["FRAIS","Normal"]
            frais_df = frais_anom.merge(frais_norm, on="FRAIS")
            fig = go.Figure(data=[
                go.Bar(name="Normal",  x=frais_df["FRAIS"], y=frais_df["Normal"],
                       marker_color="#3fb950", opacity=0.85),
                go.Bar(name="Anomaly", x=frais_df["FRAIS"], y=frais_df["Anomaly"],
                       marker_color="#f85149", opacity=0.85),
            ])
            fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=320,
                              title="Normal vs Anomaly by Fee Type")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Amount Distribution (CV EN TND)</div>', unsafe_allow_html=True)
    if "CV EN TND" in df.columns and "anomaly_IF" in df.columns:
        normal_amounts = df[df["anomaly_IF"] == 1]["CV EN TND"].clip(upper=df["CV EN TND"].quantile(0.995))
        anom_amounts   = df[df["anomaly_IF"] == -1]["CV EN TND"].clip(upper=df["CV EN TND"].quantile(0.995))

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=normal_amounts, name="Normal",
                                   marker_color="#3fb950", opacity=0.65,
                                   nbinsx=80, histnorm="probability density"))
        fig.add_trace(go.Histogram(x=anom_amounts,   name="Anomaly",
                                   marker_color="#f85149", opacity=0.75,
                                   nbinsx=80, histnorm="probability density"))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=320,
                          title="Amount Distribution — Normal vs Anomaly",
                          xaxis_title="CV EN TND", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — ANOMALY ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown('<div class="section-title">Anomaly Score Distribution</div>', unsafe_allow_html=True)
        if "anomaly_score_IF" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df["anomaly_IF"] == 1]["anomaly_score_IF"],
                name="Normal", marker_color="#3fb950", opacity=0.7,
                nbinsx=60, histnorm="probability density"))
            fig.add_trace(go.Histogram(
                x=df[df["anomaly_IF"] == -1]["anomaly_score_IF"],
                name="Anomaly", marker_color="#f85149", opacity=0.8,
                nbinsx=60, histnorm="probability density"))
            fig.add_vline(x=0, line_dash="dash", line_color="#c9d1d9", line_width=1.5,
                          annotation_text="Decision boundary (0)",
                          annotation_font_color="#8b949e")
            fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=320,
                              title="IF Anomaly Score Distribution",
                              xaxis_title="Score (lower = more suspicious)",
                              yaxis_title="Density")
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Score Summary</div>', unsafe_allow_html=True)
        if "anomaly_score_IF" in df.columns:
            score_stats = df[df["anomaly_IF"] == -1]["anomaly_score_IF"].describe()
            stats_df = pd.DataFrame({
                "Metric": ["Count","Min score","Mean score","Max score","Std"],
                "Value":  [
                    f"{int(score_stats['count']):,}",
                    f"{score_stats['min']:.4f}",
                    f"{score_stats['mean']:.4f}",
                    f"{score_stats['max']:.4f}",
                    f"{score_stats['std']:.4f}",
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

    # AGE column removed (it was branch code, not customer age)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Method Comparison — IF vs Mahalanobis</div>',
                    unsafe_allow_html=True)
        if "anomaly_Distance" in df.columns:
            ct = pd.crosstab(df["anomaly_IF"], df["anomaly_Distance"],
                             rownames=["IF"], colnames=["Distance"])
            ct.index = ct.index.map({1:"Normal (IF)", -1:"Anomaly (IF)"})
            ct.columns = ct.columns.map({1:"Normal (Dist)", -1:"Anomaly (Dist)"})
            fig = px.imshow(ct, text_auto=True, color_continuous_scale="Blues",
                            title="Confusion Matrix: IF vs Distance Method")
            fig.update_layout(**PLOTLY_LAYOUT, height=280)
            st.plotly_chart(fig, use_container_width=True)




# ═══════════════════════════════════════════════════════════════════
# TAB 3 — GEOGRAPHY
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Anomalies by Country of Origin</div>', unsafe_allow_html=True)

    if "Nom_pays" in df.columns and "anomaly_IF" in df.columns:
        country_stats = df.groupby("Nom_pays").agg(
            Total=("CHRONO","count"),
            Anomalies=("anomaly_IF", lambda x: (x == -1).sum()),
            Volume_TND=("CV EN TND","sum"),
        ).reset_index()
        country_stats["Anomaly_Rate_%"] = (country_stats["Anomalies"] / country_stats["Total"] * 100).round(2)
        country_stats = country_stats.sort_values("Anomalies", ascending=False)

        col_a, col_b = st.columns(2)
        with col_a:
            top15 = country_stats.head(15)
            fig = px.bar(top15, x="Anomalies", y="Nom_pays", orientation="h",
                         color="Anomaly_Rate_%",
                         color_continuous_scale=["#3fb950","#d29922","#f85149"],
                         title="Top 15 Countries — Anomaly Count & Rate",
                         labels={"Nom_pays":"Country","Anomalies":"Anomaly Count",
                                 "Anomaly_Rate_%":"Anomaly Rate (%)"})
            fig.update_layout(**PLOTLY_LAYOUT, height=450)
            fig.update_yaxes(autorange="reversed", gridcolor="#21262d")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">Country Stats Table</div>', unsafe_allow_html=True)
            display_cs = country_stats.copy()
            display_cs["Volume_TND"] = display_cs["Volume_TND"].map("{:,.0f}".format)
            st.dataframe(display_cs, hide_index=True, use_container_width=True, height=400)

        # Volume by country for anomalies only
        st.markdown('<div class="section-title">Suspicious Volume by Country</div>', unsafe_allow_html=True)
        anom_country = (df[df["anomaly_IF"] == -1]
                        .groupby("Nom_pays")["CV EN TND"].sum()
                        .reset_index()
                        .sort_values("CV EN TND", ascending=False)
                        .head(15))
        anom_country.columns = ["Country","Volume_TND"]
        fig = px.treemap(anom_country, path=["Country"], values="Volume_TND",
                         title="Suspicious Transaction Volume by Country (TND)",
                         color="Volume_TND",
                         color_continuous_scale=["#1f6feb","#f85149"])
        fig.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig, use_container_width=True)


# Transaction table moved to sidebar export
@st.cache_data
def to_csv(df_):
    return df_.to_csv(index=False).encode("utf-8")

with st.sidebar:
    st.download_button("⬇ Download filtered CSV", data=to_csv(df_display),
                       file_name="fraud_detection_results.csv", mime="text/csv",
                       use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — MODEL EVALUATION & INSIGHTS
# ═══════════════════════════════════════════════════════════════════
with tab4:

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART A — MODEL EVALUATION METRICS (comes FIRST)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("""
    <div class="alert-blue">
      <h3 style="margin:0;color:#58a6ff">📏 Model Evaluation Metrics</h3>
      <p style="margin-top:8px">These metrics evaluate the <b>quality of the Isolation Forest model itself</b> — 
      not the fraud results. In unsupervised learning (no labeled fraud data), we use specialized 
      metrics to answer: <i>"Is the model finding real patterns, or just noise?"</i></p>
    </div>
    """, unsafe_allow_html=True)

    try:
        _art = joblib.load("fraud_model.pkl")
        em = _art.get("evaluation_metrics", {})
    except Exception:
        em = {}

    if em:
        # ── Row 1: Core Clustering Metrics ──
        st.markdown('<div class="section-title">1️⃣ Internal Clustering Metrics</div>', unsafe_allow_html=True)
        st.markdown("These measure how well the model separates anomalies from normal transactions **without any ground truth labels**.")

        m1, m2, m3 = st.columns(3)
        with m1:
            sil = em.get("silhouette", 0)
            interp = "Excellent" if sil > 0.5 else "Good" if sil > 0.25 else "Moderate" if sil > 0.1 else "Weak"
            color = "#3fb950" if sil > 0.25 else "#d29922" if sil > 0.1 else "#f85149"
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Silhouette Score</div>
              <div class="kpi-value" style="color:{color}">{sil:.4f}</div>
              <div class="kpi-delta" style="color:{color}">{interp} — Range: [-1, +1]</div>
            </div>""", unsafe_allow_html=True)
            st.caption("**What**: Measures cluster cohesion vs separation. Higher = anomalies are clearly distinct from normal transactions. **Why**: Standard metric for unsupervised learning — confirms the model finds meaningful groupings, not random noise.")

        with m2:
            db = em.get("davies_bouldin", 0)
            interp = "Excellent" if db < 1.0 else "Good" if db < 2.0 else "Moderate"
            color = "#3fb950" if db < 1.0 else "#d29922" if db < 2.0 else "#f85149"
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Davies-Bouldin Index</div>
              <div class="kpi-value" style="color:{color}">{db:.4f}</div>
              <div class="kpi-delta" style="color:{color}">{interp} — Lower is better</div>
            </div>""", unsafe_allow_html=True)
            st.caption("**What**: Ratio of within-cluster scatter to between-cluster separation. Lower = tighter, more separated groups. **Why**: Complements Silhouette — penalizes overlapping clusters where anomalies blend with normal data.")

        with m3:
            ch = em.get("calinski_harabasz", 0)
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Calinski-Harabasz Index</div>
              <div class="kpi-value" style="color:#58a6ff">{ch:.1f}</div>
              <div class="kpi-delta" style="color:#58a6ff">Higher is better</div>
            </div>""", unsafe_allow_html=True)
            st.caption("**What**: Ratio of between-cluster variance to within-cluster variance. Higher = denser clusters, better defined. **Why**: Measures overall cluster quality — high values mean anomalies form a distinct, compact group.")

        # ── Row 2: Score Analysis ──
        st.markdown('<div class="section-title">2️⃣ Score Distribution Analysis</div>', unsafe_allow_html=True)
        st.markdown("Analyzes how well the Isolation Forest anomaly scores separate the two classes.")

        m1, m2 = st.columns(2)
        with m1:
            sep = em.get("score_separation", 0)
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Score Separation (|Mean Difference|)</div>
              <div class="kpi-value" style="color:#bc8cff">{sep:.5f}</div>
              <div class="kpi-delta" style="color:#8b949e">Normal mean: {em.get('norm_score_mean',0):.5f} | Anomaly mean: {em.get('anom_score_mean',0):.5f}</div>
            </div>""", unsafe_allow_html=True)
            st.caption("**What**: Absolute difference between average anomaly score and average normal score. **Why**: Larger separation means the model assigns clearly different scores to the two groups — easier to set a decision threshold.")

        with m2:
            agr = em.get("method_agreement", 0)
            hc = em.get("n_high_confidence", 0)
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Method Agreement (IF vs Mahalanobis)</div>
              <div class="kpi-value" style="color:#3fb950">{agr:.1f}%</div>
              <div class="kpi-delta" style="color:#8b949e">High confidence (both): {hc:,} | Only IF: {em.get('only_if',0):,} | Only Maha: {em.get('only_dist',0):,}</div>
            </div>""", unsafe_allow_html=True)
            st.caption("**What**: % of transactions where both methods agree. **Why**: Two independent algorithms reaching the same conclusion is a strong validation signal. High agreement = robust detection, not algorithm-specific artifacts.")

        # ── Row 3: Stability & Sensitivity ──
        st.markdown('<div class="section-title">3️⃣ Model Robustness</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            cv = em.get("stability_cv", 0)
            runs = em.get("stability_runs", [])
            interp = "Very Stable" if cv < 1 else "Stable" if cv < 5 else "Variable"
            color = "#3fb950" if cv < 5 else "#d29922"
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Model Stability (CV across 5 seeds)</div>
              <div class="kpi-value" style="color:{color}">{cv:.2f}%</div>
              <div class="kpi-delta" style="color:{color}">{interp} — Runs: {runs}</div>
            </div>""", unsafe_allow_html=True)
            st.caption("**What**: Coefficient of Variation of anomaly counts when re-training with 5 different random seeds. **Why**: CV < 5% means the model gives consistent results regardless of randomness — critical for production reliability.")

        with m2:
            contam = em.get("contamination_sensitivity", {})
            if contam:
                rates = [float(k) for k in contam.keys()]
                counts = list(contam.values())
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"{r:.0%}" for r in rates], y=counts,
                    marker_color=["#d29922" if f"{r}" != str(em.get("contamination", 0.05)) else "#f85149" for r in rates],
                    text=counts, textposition="auto"))
                fig.update_layout(**PLOTLY_LAYOUT, height=250,
                    title="Contamination Sensitivity Analysis",
                    xaxis_title="Contamination Rate", yaxis_title="Anomalies Detected")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**What**: Number of anomalies detected at different contamination rates. **Why**: Shows how sensitive the model is to this hyperparameter. The red bar is the selected rate (5%). A smooth curve indicates stable behavior.")

        # ── Row 4: Statistical Validation ──
        st.markdown('<div class="section-title">4️⃣ Statistical Validation</div>', unsafe_allow_html=True)

        mw = em.get("mann_whitney", {})
        if mw:
            st.markdown("**Mann-Whitney U Test** — Tests whether anomalies are statistically different from normal transactions (H₀: no difference).")
            mw_rows = []
            for col, vals in mw.items():
                p = vals["p"]
                sig = "✅ SIGNIFICANT" if p < 0.05 else "❌ Not significant"
                ratio = vals["anom_mean"] / vals["norm_mean"] if vals["norm_mean"] > 0 else 0
                mw_rows.append({"Feature": col, "p-value": f"{p:.2e}", "Result": sig,
                                "Anomaly Mean": f"{vals['anom_mean']:,.0f}",
                                "Normal Mean": f"{vals['norm_mean']:,.0f}",
                                "Ratio": f"{ratio:.1f}x"})
            st.dataframe(pd.DataFrame(mw_rows), hide_index=True, use_container_width=True)
            st.caption("**Why**: If p < 0.05, anomalies are genuinely different from normal transactions — not random noise. A high ratio (e.g., 27x) means anomalies involve amounts 27 times larger on average.")

        # ── Row 5: Proxy Validation ──
        roc = em.get("roc_auc_proxy")
        pr = em.get("pr_auc_proxy")
        if roc is not None or pr is not None:
            st.markdown('<div class="section-title">5️⃣ Proxy Validation (vs ETAT_RPA KO flags)</div>', unsafe_allow_html=True)
            st.markdown("Using RPA rejection flags (KO) as an **approximate ground truth** to estimate supervised metrics.")
            m1, m2 = st.columns(2)
            with m1:
                if roc is not None:
                    interp = "Excellent" if roc > 0.9 else "Good" if roc > 0.7 else "Moderate" if roc > 0.6 else "Weak"
                    color = "#3fb950" if roc > 0.7 else "#d29922"
                    st.markdown(f"""
                    <div class="kpi-card">
                      <div class="kpi-label">ROC-AUC (vs KO proxy)</div>
                      <div class="kpi-value" style="color:{color}">{roc:.4f}</div>
                      <div class="kpi-delta" style="color:{color}">{interp} — 0.5=random, 1.0=perfect</div>
                    </div>""", unsafe_allow_html=True)
                    st.caption("**What**: Area Under the ROC Curve — measures discriminative power across all thresholds. **Why**: Shows the model's ability to rank true positives above false positives. > 0.7 = good detection power.")
            with m2:
                if pr is not None:
                    st.markdown(f"""
                    <div class="kpi-card">
                      <div class="kpi-label">PR-AUC (Precision-Recall)</div>
                      <div class="kpi-value" style="color:#d29922">{pr:.4f}</div>
                      <div class="kpi-delta" style="color:#8b949e">Higher = better alert quality</div>
                    </div>""", unsafe_allow_html=True)
                    st.caption("**What**: Area Under the Precision-Recall Curve — critical when fraud is rare. **Why**: Unlike ROC-AUC, PR-AUC focuses on the quality of positive predictions. High PR-AUC = fewer false alarms sent to compliance team.")

        # ── Summary Table ──
        st.markdown('<div class="section-title">📋 Complete Metrics Summary</div>', unsafe_allow_html=True)
        summary = [
            ["Silhouette Score", f"{em.get('silhouette',0):.4f}", "[-1, +1]", "Cluster separation quality"],
            ["Davies-Bouldin", f"{em.get('davies_bouldin',0):.4f}", "≥ 0 (lower=better)", "Cluster compactness"],
            ["Calinski-Harabasz", f"{em.get('calinski_harabasz',0):.1f}", "≥ 0 (higher=better)", "Cluster density ratio"],
            ["Score Separation", f"{em.get('score_separation',0):.5f}", "≥ 0 (higher=better)", "IF score class gap"],
            ["Method Agreement", f"{em.get('method_agreement',0):.1f}%", "0-100%", "IF vs Mahalanobis consensus"],
            ["Stability CV", f"{em.get('stability_cv',0):.2f}%", "< 5% = stable", "Reproducibility across seeds"],
            ["ROC-AUC (proxy)", f"{em.get('roc_auc_proxy','N/A')}" if em.get('roc_auc_proxy') is None else f"{em.get('roc_auc_proxy',0):.4f}", "0.5-1.0", "Discriminative power"],
            ["PR-AUC (proxy)", f"{em.get('pr_auc_proxy','N/A')}" if em.get('pr_auc_proxy') is None else f"{em.get('pr_auc_proxy',0):.4f}", "0-1.0", "Alert precision quality"],
        ]
        st.dataframe(pd.DataFrame(summary, columns=["Metric", "Value", "Range", "Description"]),
                      hide_index=True, use_container_width=True)

    else:
        st.markdown("""
        <div class="alert-orange">
          <b>⚠ Evaluation metrics not found.</b><br>
          Please re-run <code>python -X utf8 main.py</code> to compute and save metrics to <code>fraud_model.pkl</code>.
        </div>
        """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART B — HOW THE MODEL WORKS (educational diagrams with REAL data)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown('<div class="section-title">🧠 How the Model Works — Your Real Data</div>', unsafe_allow_html=True)

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Ellipse as MplEllipse
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _SS

    _numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    _exclude = ["anomaly_IF", "anomaly_score_IF", "mahalanobis_distance", "high_confidence_fraud"]
    _viz_cols = [c for c in _numeric_cols if c not in _exclude]

    _has_viz = False
    if len(_viz_cols) >= 2:
        _X = df[_viz_cols].fillna(0).values
        _X_scaled = _SS().fit_transform(_X)
        _X_pca = _PCA(n_components=2, random_state=42).fit_transform(_X_scaled)

        _labels = df["anomaly_IF"].values
        _maha = df["mahalanobis_distance"].values if "mahalanobis_distance" in df.columns else np.zeros(len(df))
        _has_viz = True

    col_if, col_maha = st.columns(2)

    # ─────────────────────────────────────────────────────────────
    # CHART 1 — Isolation Forest
    # ─────────────────────────────────────────────────────────────
    with col_if:
        st.markdown("""
        <div class="alert-blue">
          <h3 style="margin:0;color:#58a6ff">🌲 Isolation Forest</h3>
          <p style="margin-top:5px"><b>The Primary Detector</b></p>
        </div>
        """, unsafe_allow_html=True)

        if _has_viz:
            fig_if, ax_if = plt.subplots(figsize=(6, 5), facecolor="#0d1117")
            ax_if.set_facecolor("#0d1117")

            # Plot normal points
            normal_mask = _labels == 1
            anomaly_mask = _labels == -1
            ax_if.scatter(_X_pca[normal_mask, 0], _X_pca[normal_mask, 1],
                          c="#3fb950", s=6, alpha=0.4, label=f"Normal ({normal_mask.sum():,})", zorder=2)
            ax_if.scatter(_X_pca[anomaly_mask, 0], _X_pca[anomaly_mask, 1],
                          c="#f85149", s=12, alpha=0.8, label=f"Anomaly ({anomaly_mask.sum():,})", zorder=3, edgecolors="#ff7b72", linewidths=0.3)

            # Draw a boundary ellipse around the normal cluster
            _mean_n = _X_pca[normal_mask].mean(axis=0)
            _cov_n = np.cov(_X_pca[normal_mask].T)
            _eigvals, _eigvecs = np.linalg.eigh(_cov_n)
            _angle = np.degrees(np.arctan2(_eigvecs[1, 1], _eigvecs[0, 1]))
            for _n_std, _color, _lw, _ls, _lbl in [
                (2.0, "#3fb950", 2, "--", "Normal Zone"),
                (3.0, "#d29922", 2, ":", "Border Zone"),
            ]:
                _w = 2 * _n_std * np.sqrt(_eigvals[0])
                _h = 2 * _n_std * np.sqrt(_eigvals[1])
                ell = MplEllipse(xy=_mean_n, width=_w, height=_h, angle=_angle,
                                 edgecolor=_color, facecolor="none", linewidth=_lw, linestyle=_ls, zorder=4)
                ax_if.add_patch(ell)

            # Annotations
            ax_if.annotate("Dense Normal\nCluster", xy=_mean_n, fontsize=8, color="#8b949e",
                           ha="center", va="center", zorder=5,
                           bbox=dict(boxstyle="round,pad=0.3", fc="#161b22", ec="#30363d", alpha=0.8))
            
            # Find an anomaly point for annotation
            _anom_pts = _X_pca[anomaly_mask]
            if len(_anom_pts) > 0:
                _far_idx = np.argmax(np.linalg.norm(_anom_pts - _mean_n, axis=1))
                _far_pt = _anom_pts[_far_idx]
                ax_if.annotate("Isolated\nAnomaly", xy=_far_pt, fontsize=7, color="#f85149",
                               ha="center", va="bottom", xytext=(_far_pt[0], _far_pt[1] + 1),
                               arrowprops=dict(arrowstyle="->", color="#f85149", lw=1),
                               bbox=dict(boxstyle="round,pad=0.2", fc="#161b22", ec="#f85149", alpha=0.8), zorder=5)

            ax_if.set_title("Isolation Forest — Your 13,004 Transactions", color="#c9d1d9", fontsize=10, fontweight="bold", pad=10)
            ax_if.legend(loc="upper right", fontsize=7, facecolor="#161b22", edgecolor="#30363d",
                         labelcolor="#c9d1d9", framealpha=0.9)
            ax_if.set_xticks([])
            ax_if.set_yticks([])
            for spine in ax_if.spines.values():
                spine.set_color("#30363d")
            plt.tight_layout()
            st.pyplot(fig_if)
            plt.close(fig_if)

        st.markdown("""
        **Simple Explanation:**  
        - The **green dots** are your normal banking transactions — packed tightly together.  
        - The **red dots** are anomalies — isolated far from the main group.  
        - The **dashed ellipses** show the boundary: anything outside stands out.
        """)

    # ─────────────────────────────────────────────────────────────
    # CHART 2 — Mahalanobis Distance
    # ─────────────────────────────────────────────────────────────
    with col_maha:
        st.markdown("""
        <div class="alert-blue">
          <h3 style="margin:0;color:#58a6ff">📐 Mahalanobis Distance</h3>
          <p style="margin-top:5px"><b>The Statistical Validator</b></p>
        </div>
        """, unsafe_allow_html=True)

        if _has_viz:
            fig_m, ax_m = plt.subplots(figsize=(6, 5), facecolor="#0d1117")
            ax_m.set_facecolor("#0d1117")

            # Thresholds
            _p90 = np.percentile(_maha, 90)
            _p95 = np.percentile(_maha, 95)

            # Color each point by zone
            _green = _maha <= _p90
            _yellow = (_maha > _p90) & (_maha <= _p95)
            _red = _maha > _p95

            ax_m.scatter(_X_pca[_green, 0], _X_pca[_green, 1],
                         c="#3fb950", s=6, alpha=0.4, label=f"Normal ({_green.sum():,})", zorder=2)
            ax_m.scatter(_X_pca[_yellow, 0], _X_pca[_yellow, 1],
                         c="#d29922", s=10, alpha=0.6, label=f"Watch ({_yellow.sum():,})", zorder=3)
            ax_m.scatter(_X_pca[_red, 0], _X_pca[_red, 1],
                         c="#f85149", s=14, alpha=0.8, label=f"Anomaly ({_red.sum():,})", zorder=3, edgecolors="#ff7b72", linewidths=0.3)

            # Draw proper statistical confidence ellipses from the covariance
            _mean_all = _X_pca.mean(axis=0)
            _cov_all = np.cov(_X_pca.T)
            _eigvals_a, _eigvecs_a = np.linalg.eigh(_cov_all)
            _angle_a = np.degrees(np.arctan2(_eigvecs_a[1, 1], _eigvecs_a[0, 1]))

            for _n_std, _color, _lw, _ls, _lbl in [
                (1.5, "#3fb950", 2.5, "-",  "Normal Zone (90%)"),
                (2.5, "#d29922", 2.0, "--", "Watch Zone (95%)"),
                (3.5, "#f85149", 1.5, ":",  "Anomaly Zone"),
            ]:
                _w = 2 * _n_std * np.sqrt(_eigvals_a[0])
                _h = 2 * _n_std * np.sqrt(_eigvals_a[1])
                ell = MplEllipse(xy=_mean_all, width=_w, height=_h, angle=_angle_a,
                                 edgecolor=_color, facecolor="none", linewidth=_lw, linestyle=_ls, zorder=4)
                ax_m.add_patch(ell)

            # Center marker
            ax_m.plot(_mean_all[0], _mean_all[1], marker="+", color="#58a6ff", markersize=14, markeredgewidth=2, zorder=5)
            ax_m.annotate("Statistical\nCenter", xy=_mean_all, fontsize=8, color="#58a6ff",
                          ha="center", va="top", xytext=(_mean_all[0], _mean_all[1] - 0.8),
                          bbox=dict(boxstyle="round,pad=0.3", fc="#161b22", ec="#58a6ff", alpha=0.8), zorder=5)

            # Label the zones
            ax_m.annotate("Normal Zone", xy=(_mean_all[0] + 1.5*np.sqrt(_eigvals_a[0]), _mean_all[1]),
                          fontsize=7, color="#3fb950",
                          bbox=dict(boxstyle="round,pad=0.2", fc="#161b22", ec="#3fb950", alpha=0.7), zorder=5)
            ax_m.annotate("Watch Zone", xy=(_mean_all[0] + 2.5*np.sqrt(_eigvals_a[0]), _mean_all[1] + 0.5),
                          fontsize=7, color="#d29922",
                          bbox=dict(boxstyle="round,pad=0.2", fc="#161b22", ec="#d29922", alpha=0.7), zorder=5)

            ax_m.set_title("Mahalanobis Distance — Your 13,004 Transactions", color="#c9d1d9", fontsize=10, fontweight="bold", pad=10)
            ax_m.legend(loc="upper right", fontsize=7, facecolor="#161b22", edgecolor="#30363d",
                        labelcolor="#c9d1d9", framealpha=0.9)
            ax_m.set_xticks([])
            ax_m.set_yticks([])
            for spine in ax_m.spines.values():
                spine.set_color("#30363d")
            plt.tight_layout()
            st.pyplot(fig_m)
            plt.close(fig_m)

        st.markdown("""
        **Simple Explanation:**  
        - The **blue cross** is the statistical center of all your transactions.  
        - The **ellipses** are the boundaries: green = normal, yellow = watch, red = danger.  
        - Transactions landing **outside the ellipses** are flagged as anomalies.
        """)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CROSS-MODEL INTERSECTION (TABLEAU CROISÉ)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    st.markdown('<div class="section-title">🤝 Interprétation du Tableau Croisé (Intersection des Modèles)</div>', unsafe_allow_html=True)

    if "mahalanobis_distance" in df.columns and "anomaly_IF" in df.columns:
        # Dynamically compute thresholds based on the dataset
        threshold_m = df["mahalanobis_distance"].quantile(0.95)
        mahal_anom = df["mahalanobis_distance"] > threshold_m
        if_anom = df["anomaly_IF"] == -1
        
        both_anom = (mahal_anom & if_anom).sum()
        only_if = (~mahal_anom & if_anom).sum()
        only_mahal = (mahal_anom & ~if_anom).sum()
        both_normal = (~mahal_anom & ~if_anom).sum()

        # ── Tableau Croisé (Plotly Heatmap avec Couleurs Personnalisées) ──
        import plotly.graph_objects as go
        
        x_labels = ['🔴 Anomalie (IF)', '🟢 Normal (IF)']
        y_labels = ['🔴 Anomalie (Mahal)', '🟢 Normal (Mahal)']
        
        # Color mapping indices:
        # 0 = Red (Both Anom)
        # 1 = Yellow (Mahal Only)
        # 2 = Orange (IF Only)
        # 3 = Green (Both Normal)
        z_color = [[0, 1],
                   [2, 3]]
        
        text_matrix = [[f"{int(both_anom):,}", f"{int(only_mahal):,}"],
                       [f"{int(only_if):,}", f"{int(both_normal):,}"]]
        
        custom_colors = [
            [0.0, 'rgba(248, 81, 73, 0.8)'], [0.25, 'rgba(248, 81, 73, 0.8)'], # Red
            [0.25, 'rgba(210, 153, 34, 0.8)'], [0.5, 'rgba(210, 153, 34, 0.8)'],   # Yellow
            [0.5, 'rgba(255, 123, 114, 0.8)'], [0.75, 'rgba(255, 123, 114, 0.8)'], # Orange
            [0.75, 'rgba(63, 185, 80, 0.8)'], [1.0, 'rgba(63, 185, 80, 0.8)']      # Green
        ]
        
        fig_cross = go.Figure(data=go.Heatmap(
            z=z_color,
            x=x_labels,
            y=y_labels,
            text=text_matrix,
            texttemplate="<b>%{text}</b>",
            textfont={"size": 26, "color": "white"},
            colorscale=custom_colors,
            showscale=False,
            hoverinfo="skip",
            xgap=3, ygap=3 # Adds a nice small gap between cells
        ))
        
        fig_cross.update_layout(
            title_text="<b>Matrice d'Intersection des Modèles</b>",
            title_x=0.5,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9", size=14),
            margin=dict(t=60, l=150, b=50, r=50),
            height=350,
            yaxis=dict(autorange="reversed") # Keeps Anomalie Mahal on the top row
        )

        st.plotly_chart(fig_cross, use_container_width=True)

        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 25px;">
            <div style="background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 4px solid #f85149;">
                <h4 style="margin-top: 0; color: #f85149;">🔴 Anomalie Haute Confiance (Les Deux)</h4>
                <h2 style="margin: 5px 0;">{both_anom:,}</h2>
                <p style="font-size: 0.85em; color: #8b949e; margin-bottom: 0;">Détectées par Mahalanobis ET Isolation Forest.</p>
            </div>
            <div style="background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 4px solid #3fb950;">
                <h4 style="margin-top: 0; color: #3fb950;">🟢 Transactions Normales (Les Deux)</h4>
                <h2 style="margin: 5px 0;">{both_normal:,}</h2>
                <p style="font-size: 0.85em; color: #8b949e; margin-bottom: 0;">Considérées normales par les deux modèles.</p>
            </div>
            <div style="background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 4px solid #ff7b72;">
                <h4 style="margin-top: 0; color: #ff7b72;">🟠 Anomalie Isolation Forest Uniquement</h4>
                <h2 style="margin: 5px 0;">{only_if:,}</h2>
                <p style="font-size: 0.85em; color: #8b949e; margin-bottom: 0;">Structures atypiques non linéaires.</p>
            </div>
            <div style="background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 4px solid #d29922;">
                <h4 style="margin-top: 0; color: #d29922;">🟡 Anomalie Mahalanobis Uniquement</h4>
                <h2 style="margin: 5px 0;">{only_mahal:,}</h2>
                <p style="font-size: 0.85em; color: #8b949e; margin-bottom: 0;">Forte distance statistique, mais cohérence locale.</p>
            </div>
        </div>

        <div style="background-color: #0d1117; padding: 20px; border-radius: 8px; border: 1px solid #30363d;">
            <p><b>🔴 {both_anom:,} transactions détectées par les deux méthodes :</b><br>
            <span style="color: #c9d1d9;">Ces observations représentent les cas les plus critiques. Le fait qu’elles soient simultanément détectées par deux approches indépendantes renforce fortement leur caractère atypique. Elles constituent donc des anomalies de haute confiance, devant être considérées comme prioritaires dans une démarche d’investigation ou d’audit.</span></p>
            
            <p><b>🟠 {only_if:,} transactions détectées uniquement par Isolation Forest :</b><br>
            <span style="color: #c9d1d9;">Ces observations peuvent correspondre à des comportements complexes ou non linéaires. Isolation Forest étant capable de détecter des structures atypiques dans des espaces multidimensionnels complexes, certaines transactions peuvent apparaître inhabituelles dans la structure des arbres sans être nécessairement très éloignées du centre statistique global.</span></p>
            
            <p><b>🟡 {only_mahal:,} transactions détectées uniquement par Mahalanobis :</b><br>
            <span style="color: #c9d1d9;">Ces transactions présentent une forte distance statistique par rapport au centre des données. Cependant, leur structure locale peut rester relativement cohérente dans l’espace analysé, ce qui explique pourquoi Isolation Forest ne les considère pas forcément comme isolées.</span></p>
            
            <p><b>🟢 {both_normal:,} transactions considérées normales par les deux méthodes :</b><br>
            <span style="color: #c9d1d9;">Cette forte majorité confirme l’existence d’un noyau principal de comportements homogènes et statistiquement cohérents au sein du jeu de données.</span></p>
        </div>
        """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART C — DETECTION RESULTS (feature-level analysis, separate from metrics)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    st.markdown("""
    <div class="alert-green">
      <h3 style="margin:0;color:#3fb950">📊 Detection Results — Feature Analysis</h3>
      <p style="margin-top:8px">The charts below show the <b>detection results</b> — how flagged anomalies 
      differ from normal transactions on key features. These are <b>not model metrics</b>; they visualize 
      <i>what the model found</i>.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Expand Detection Results & Feature Analysis", expanded=False):

        col_a, col_b = st.columns(2)
        with col_a:
            if "mahalanobis_distance" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[df["anomaly_IF"] == 1]["mahalanobis_distance"],
                    name="Normal", marker_color="#3fb950", opacity=0.7,
                    nbinsx=60, histnorm="probability density"))
                fig.add_trace(go.Histogram(
                    x=df[df["anomaly_IF"] == -1]["mahalanobis_distance"],
                    name="Anomaly", marker_color="#f85149", opacity=0.8,
                    nbinsx=60, histnorm="probability density"))
                fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=320,
                                  title="Mahalanobis Distance Distribution — Normal vs Anomaly",
                                  xaxis_title="Statistical Distance from Center",
                                  yaxis_title="Density")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Anomalies have higher Mahalanobis distances — they are statistically far from the 'center' of normal behavior.")

        with col_b:
            if "anomaly_score_IF" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Box(y=df[df["anomaly_IF"] == 1]["anomaly_score_IF"],
                                     name="Normal", marker_color="#3fb950"))
                fig.add_trace(go.Box(y=df[df["anomaly_IF"] == -1]["anomaly_score_IF"],
                                     name="Anomaly", marker_color="#f85149"))
                fig.update_layout(**PLOTLY_LAYOUT, height=320,
                                  title="Isolation Forest Score Distribution",
                                  yaxis_title="Anomaly Score (lower = more suspicious)")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Normal transactions cluster above 0, anomalies below 0. The gap between distributions confirms good separation.")

        # Processing delay
        if "D_EXEC" in df.columns and "Date" in df.columns:
            df["processing_delay"] = (pd.to_datetime(df["D_EXEC"]) - pd.to_datetime(df["Date"])).dt.days.clip(lower=0)
        delay_col = "processing_delay" if "processing_delay" in df.columns else \
                    "processing_delay_days" if "processing_delay_days" in df.columns else None
        if delay_col:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df["anomaly_IF"] == 1][delay_col],
                name="Normal", marker_color="#3fb950", opacity=0.7,
                nbinsx=60, histnorm="probability density"))
            fig.add_trace(go.Histogram(
                x=df[df["anomaly_IF"] == -1][delay_col],
                name="Anomaly", marker_color="#f85149", opacity=0.8,
                nbinsx=60, histnorm="probability density"))
            fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=320,
                              title="Processing Delay — Normal vs Anomaly",
                              xaxis_title="Days (D_EXEC - Date)", yaxis_title="Density")
            st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if "txn_count_sender_7d" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Box(y=df[df["anomaly_IF"] == 1]["txn_count_sender_7d"],
                                     name="Normal", marker_color="#3fb950"))
                fig.add_trace(go.Box(y=df[df["anomaly_IF"] == -1]["txn_count_sender_7d"],
                                     name="Anomaly", marker_color="#f85149"))
                fig.update_layout(**PLOTLY_LAYOUT, height=320,
                                  title="Sender Velocity — 7-Day Transaction Count",
                                  yaxis_title="Transactions in 7 days")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Higher velocity in anomalies suggests burst-transfer patterns (smurfing).")

        with col_b:
            if "amount_vs_sender_avg" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Box(y=df[df["anomaly_IF"] == 1]["amount_vs_sender_avg"].clip(upper=20),
                                     name="Normal", marker_color="#3fb950"))
                fig.add_trace(go.Box(y=df[df["anomaly_IF"] == -1]["amount_vs_sender_avg"].clip(upper=20),
                                     name="Anomaly", marker_color="#f85149"))
                fig.update_layout(**PLOTLY_LAYOUT, height=320,
                                  title="Amount vs Sender Historical Average (Ratio)",
                                  yaxis_title="Ratio (1.0 = typical amount)")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Anomalies show much higher ratios — amounts far exceeding what's normal for that sender.")

        col_a, col_b = st.columns(2)
        with col_a:
            if "country_mismatch_risk" in df.columns:
                mismatch_data = pd.DataFrame({
                    "Status": ["Normal", "Anomaly"],
                    "Mismatch": [
                        int((df[df["anomaly_IF"] == 1]["country_mismatch_risk"] == 1).sum()),
                        int((df[df["anomaly_IF"] == -1]["country_mismatch_risk"] == 1).sum()),
                    ],
                    "No_Mismatch": [
                        int((df[df["anomaly_IF"] == 1]["country_mismatch_risk"] == 0).sum()),
                        int((df[df["anomaly_IF"] == -1]["country_mismatch_risk"] == 0).sum()),
                    ]
                })
                fig = go.Figure(data=[
                    go.Bar(name="No Mismatch", x=mismatch_data["Status"], y=mismatch_data["No_Mismatch"],
                           marker_color="#3fb950", opacity=0.85),
                    go.Bar(name="Country Mismatch", x=mismatch_data["Status"], y=mismatch_data["Mismatch"],
                           marker_color="#f85149", opacity=0.85),
                ])
                fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=320,
                                  title="BIC Country vs Declared Country Mismatch")
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "is_round_amount" in df.columns:
                round_data = pd.DataFrame({
                    "Status": ["Normal", "Anomaly"],
                    "Round": [
                        int((df[df["anomaly_IF"] == 1]["is_round_amount"] == 1).sum()),
                        int((df[df["anomaly_IF"] == -1]["is_round_amount"] == 1).sum()),
                    ],
                    "Not_Round": [
                        int((df[df["anomaly_IF"] == 1]["is_round_amount"] == 0).sum()),
                        int((df[df["anomaly_IF"] == -1]["is_round_amount"] == 0).sum()),
                    ]
                })
                fig = go.Figure(data=[
                    go.Bar(name="Normal Amount", x=round_data["Status"], y=round_data["Not_Round"],
                           marker_color="#3fb950", opacity=0.85),
                    go.Bar(name="Round Amount (×1000)", x=round_data["Status"], y=round_data["Round"],
                           marker_color="#d29922", opacity=0.85),
                ])
                fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=320,
                                  title="Round Amount Flag — Multiples of 1,000")
                st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_cols = [c for c in ["MNT","CV EN TND","anomaly_score_IF",
                                  "mahalanobis_distance","high_confidence_fraud",
                                  "amount_vs_sender_avg","txn_count_sender_7d",
                                  "country_mismatch_risk","is_round_amount"]
                     if c in numeric_cols]
        if len(corr_cols) >= 3:
            corr = df[corr_cols].corr().round(3)
            fig = px.imshow(corr, text_auto=True,
                            color_continuous_scale=["#f85149","#161b22","#3fb950"],
                            zmin=-1, zmax=1,
                            title="Feature Correlation Matrix")
            fig.update_layout(**PLOTLY_LAYOUT, height=420)
            st.plotly_chart(fig, use_container_width=True)



# ═══════════════════════════════════════════════════════════════════
# TAB 5 — SIMULATOR
# ═══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🧪 Simulateur de Prédiction en Temps Réel</div>', unsafe_allow_html=True)
    st.markdown("Testez le modèle en modifiant manuellement les caractéristiques d'une transaction.")
    
    try:
        model_artifacts = joblib.load("fraud_model.pkl")
        iso_forest = model_artifacts["iso_forest"]
        scaler = model_artifacts["scaler"]
        selector = model_artifacts["selector"]
        label_encoders = model_artifacts["label_encoders"]
        model_cols = model_artifacts["features_df_columns"]
        
        with st.form("simulator_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Informations Générales")
                mnt = st.number_input("Montant de la transaction", min_value=0.0, value=50000.0, step=1000.0)
                dev_opts = list(label_encoders["DEV"].keys()) if isinstance(label_encoders["DEV"], dict) else list(label_encoders["DEV"].classes_)
                dev = st.selectbox("Devise", options=dev_opts, index=dev_opts.index("EUR") if "EUR" in dev_opts else 0)
                
                pays_opts = list(label_encoders["52A_ISO_PAYS"].keys()) if isinstance(label_encoders["52A_ISO_PAYS"], dict) else list(label_encoders["52A_ISO_PAYS"].classes_)
                pays = st.selectbox("Pays d'origine", options=pays_opts, index=pays_opts.index("FR") if "FR" in pays_opts else 0)
                
                frais_opts = list(label_encoders["FRAIS"].keys()) if isinstance(label_encoders["FRAIS"], dict) else list(label_encoders["FRAIS"].classes_)
                frais = st.selectbox("Type de frais", options=frais_opts)
                
                # Added FC___ and 52A_EXPED for simulation
                if "FC___" in label_encoders:
                    fc_opts = list(label_encoders["FC___"].keys()) if isinstance(label_encoders["FC___"], dict) else []
                    fc = st.selectbox("Client (FC___)", options=fc_opts, index=0 if len(fc_opts) > 0 else None, help="Sélectionnez un identifiant client pour tester sa fréquence")
                else:
                    fc = None

                if "52A_EXPED" in label_encoders:
                    exped_opts = list(label_encoders["52A_EXPED"].keys()) if isinstance(label_encoders["52A_EXPED"], dict) else []
                    exped = st.selectbox("Banque Correspondante (52A_EXPED)", options=exped_opts, index=0 if len(exped_opts) > 0 else None)
                else:
                    exped = None
                # AGE removed (was branch code, not customer age)

            with col2:
                st.subheader("Patterns Comportementaux (Features)")
                txn_count_7d = st.slider("Vélocité (Nb transactions les 7 derniers jours)", min_value=0, max_value=50, value=1)
                amt_vs_avg = st.slider("Ratio Montant vs Moyenne Historique", min_value=0.1, max_value=100.0, value=1.0, help="1.0 = montant habituel. 10.0 = montant 10x plus grand que d'habitude.")
                mismatch = st.checkbox("Incohérence Pays/BIC (country_mismatch_risk)", help="Le pays du BIC ne correspond pas au pays déclaré.")
                high_risk = st.checkbox("Pays à Haut Risque (FATF)", help="Ex: Iran, Syrie, etc.")
                round_amt = st.checkbox("Montant Parfaitement Rond", help="Multiple de 1000 sans centimes.")

            submit = st.form_submit_button("Lancer la Prédiction 🔮")

        if submit:
            rates = {"USD": 3.1, "EUR": 3.4, "CAD": 2.3, "GBP": 3.9, "TND": 1.0}
            cv_en_tnd = mnt * rates.get(dev, 3.0)
            
            new_row = {}
            for col in model_cols:
                if col == "MNT": new_row[col] = mnt
                elif col == "CV EN TND": new_row[col] = cv_en_tnd
                elif col == "log_MNT": new_row[col] = np.log1p(mnt)
                elif col == "log_CV_EN_TND": new_row[col] = np.log1p(cv_en_tnd)
                # AGE removed from features
                elif col == "txn_count_sender_7d": new_row[col] = txn_count_7d
                elif col == "amount_vs_sender_avg": new_row[col] = amt_vs_avg
                elif col == "country_mismatch_risk": new_row[col] = 1 if mismatch else 0
                elif col == "high_risk_country": new_row[col] = 1 if high_risk else 0
                elif col == "is_round_amount": new_row[col] = 1 if round_amt else 0
                
                # One-Hot Encoded Features
                elif col.startswith("DEV_"): new_row[col] = 1 if col == f"DEV_{dev}" else 0
                elif col.startswith("FRAIS_"): new_row[col] = 1 if col == f"FRAIS_{frais}" else 0
                
                # Frequency Encoded Features
                elif col == "52A_ISO_PAYS_freq": new_row[col] = label_encoders["52A_ISO_PAYS"].get(pays, 0) if isinstance(label_encoders["52A_ISO_PAYS"], dict) else label_encoders["52A_ISO_PAYS"].transform([pays])[0]
                elif col == "FC____freq" and fc is not None: new_row[col] = label_encoders.get("FC___", {}).get(fc, 0)
                elif col == "52A_EXPED_freq" and exped is not None: new_row[col] = label_encoders.get("52A_EXPED", {}).get(exped, 0)
                else:
                    base_col = col.replace("_freq", "").replace("_enc", "")
                    # Try to get median from df, otherwise fallback to 0
                    if base_col in df.columns and pd.api.types.is_numeric_dtype(df[base_col]):
                        new_row[col] = df[base_col].median()
                    else:
                        new_row[col] = 0
            
            X_new = pd.DataFrame([new_row])[model_cols]
            X_scaled = scaler.transform(X_new)
            score = iso_forest.decision_function(X_scaled)[0]
            prediction = iso_forest.predict(X_scaled)[0]
            
            st.markdown("### Résultat de l'Intelligence Artificielle")
            if prediction == -1:
                st.markdown(f'''
                <div class="alert-red">
                  <h3 style="color:#f85149; margin:0">🚨 ANOMALIE DÉTECTÉE</h3>
                  <p style="margin-top:5px">Cette transaction présente un comportement fortement déviant.</p>
                  <b>Score d'anomalie : {score:.4f}</b> <i>(Seuil : < 0)</i>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="alert-green">
                  <h3 style="color:#3fb950; margin:0">✅ TRANSACTION NORMALE</h3>
                  <p style="margin-top:5px">Le comportement est cohérent avec l'historique bancaire habituel.</p>
                  <b>Score d'anomalie : {score:.4f}</b> <i>(Seuil : > 0)</i>
                </div>
                ''', unsafe_allow_html=True)
                
            with st.expander("Voir les données envoyées au modèle"):
                st.write(new_row)

    except FileNotFoundError:
        st.warning("Fichier `fraud_model.pkl` introuvable. Veuillez d'abord exécuter `main.py` pour générer le modèle.")


# Tab 7 (Synthèse PFE) merged into Tab 4 (Model Evaluation)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
  Banking Fraud Detection Dashboard &nbsp;|&nbsp; PFE — Final Year Project
  &nbsp;|&nbsp; Isolation Forest + Mahalanobis Distance &nbsp;|&nbsp; UIB Tunisia
</div>
""", unsafe_allow_html=True)
