"""
Streamlit Dashboard — Customer Risk Score (Repeat Purchase Prediction)
Author: [Nama Kamu]
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Customer Risk Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)


# ===== LOAD DATA =====
@st.cache_data
def load_data():
    """Load risk score data dengan caching."""
    # Path relative terhadap file app
    path = Path(__file__).parent.parent / 'data' / 'processed' / 'customer_risk_scores.csv'
    
    if not path.exists():
        st.error(f"❌ File tidak ditemukan: {path}")
        st.info("Jalankan notebook modeling dulu untuk export `customer_risk_scores.csv`")
        st.stop()
    
    df = pd.read_csv(path)
    return df


df = load_data()


# ===== HEADER =====
st.markdown('<div class="main-header">🛒 Customer Risk Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Prediksi Repeat Purchase — Prioritas Campaign Marketing</div>',
    unsafe_allow_html=True
)


# ===== SIDEBAR — FILTERS =====
st.sidebar.header("🔍 Filter")

# Filter risk segment
segments = ['All'] + sorted(df['risk_segment'].dropna().unique().tolist())
selected_segment = st.sidebar.selectbox("Risk Segment", segments)

# Filter risk score minimum
min_score = st.sidebar.slider(
    "Minimum Risk Score",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)

# Apply filter
df_filtered = df.copy()
if selected_segment != 'All':
    df_filtered = df_filtered[df_filtered['risk_segment'] == selected_segment]
df_filtered = df_filtered[df_filtered['risk_score'] >= min_score]

st.sidebar.markdown("---")
st.sidebar.metric("Customer setelah filter", f"{len(df_filtered):,}")


# ===== KPI METRICS =====
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Customers",
        value=f"{len(df_filtered):,}"
    )

with col2:
    high_risk = (df_filtered['risk_score'] >= 0.3).sum()
    st.metric(
        label="High Risk (≥0.3)",
        value=f"{high_risk:,}",
        delta=f"{high_risk/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%"
    )

with col3:
    avg_score = df_filtered['risk_score'].mean() if len(df_filtered) > 0 else 0
    st.metric(
        label="Avg Risk Score",
        value=f"{avg_score:.3f}"
    )

with col4:
    if 'is_repeat' in df_filtered.columns:
        actual_repeat = df_filtered['is_repeat'].mean() * 100
        st.metric(
            label="Actual Repeat Rate",
            value=f"{actual_repeat:.2f}%"
        )


# ===== CHART 1: DISTRIBUSI RISK SCORE =====
st.markdown("---")
st.subheader("📊 Distribusi Risk Score")

col_left, col_right = st.columns([2, 1])

with col_left:
    fig_hist = px.histogram(
        df_filtered,
        x='risk_score',
        nbins=50,
        color='risk_segment',
        title='Distribusi Risk Score per Segment',
        labels={'risk_score': 'Risk Score', 'count': 'Jumlah Customer'},
        color_discrete_map={
            'Low': '#2ecc71',
            'Medium': '#f39c12',
            'High': '#e67e22',
            'Very High': '#e74c3c'
        }
    )
    fig_hist.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    seg_counts = df_filtered['risk_segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    
    fig_pie = px.pie(
        seg_counts,
        values='Count',
        names='Segment',
        title='Proporsi Segment',
        color='Segment',
        color_discrete_map={
            'Low': '#2ecc71',
            'Medium': '#f39c12',
            'High': '#e67e22',
            'Very High': '#e74c3c'
        },
        hole=0.4
    )
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)


# ===== CHART 2: RISK SCORE VS ACTUAL =====
st.markdown("---")
st.subheader("🎯 Kalibrasi Model — Risk Score vs Actual Repeat")

# Buat decile
df_filtered['decile'] = pd.qcut(df_filtered['risk_score'], q=10, labels=False, duplicates='drop')

if 'is_repeat' in df_filtered.columns:
    decile_actual = df_filtered.groupby('decile').agg(
        avg_score=('risk_score', 'mean'),
        actual_repeat_rate=('is_repeat', 'mean'),
        count=('customer_unique_id', 'count')
    ).reset_index()
    decile_actual['actual_repeat_rate'] *= 100
    
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Bar(
        x=decile_actual['decile'],
        y=decile_actual['actual_repeat_rate'],
        name='Actual Repeat Rate (%)',
        marker_color='#3498db'
    ))
    fig_cal.add_trace(go.Scatter(
        x=decile_actual['decile'],
        y=decile_actual['avg_score'] * 100,
        name='Avg Predicted Score (%)',
        mode='lines+markers',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=10)
    ))
    fig_cal.update_layout(
        title='Kalibrasi: Decile Risk Score vs Actual Repeat Rate',
        xaxis_title='Decile (0=lowest risk, 9=highest risk)',
        yaxis_title='Rate (%)',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_cal, use_container_width=True)
    
    st.caption(
        "💡 **Cara baca:** Jika model bagus, decile tinggi (kanan) harus punya "
        "actual repeat rate lebih tinggi dari decile rendah (kiri). "
        "Bar biru = aktual, garis merah = prediksi."
    )


# ===== CHART 3: SCATTER =====
st.markdown("---")
st.subheader("🔍 Segmentasi Detail")

fig_scatter = px.scatter(
    df_filtered,
    x='risk_score',
    y='is_repeat' if 'is_repeat' in df_filtered.columns else 'risk_score',
    color='risk_segment',
    title='Risk Score vs Actual Repeat (jitter)',
    labels={'risk_score': 'Risk Score', 'is_repeat': 'Actual Repeat (0/No, 1/Yes)'},
    color_discrete_map={
        'Low': '#2ecc71',
        'Medium': '#f39c12',
        'High': '#e67e22',
        'Very High': '#e74c3c'
    },
    opacity=0.6
)
fig_scatter.update_layout(height=450)
st.plotly_chart(fig_scatter, use_container_width=True)


# ===== TABEL TOP CUSTOMERS =====
st.markdown("---")
st.subheader("🏆 Top Customers Paling Berisiko (Prioritas Campaign)")

top_n = st.slider("Tampilkan berapa customer?", 10, 100, 20, step=10)

top_customers = df_filtered.head(top_n).copy()
top_customers['risk_score'] = top_customers['risk_score'].round(4)
top_customers.index = range(1, len(top_customers) + 1)

st.dataframe(
    top_customers[['customer_unique_id', 'risk_score', 'risk_segment'] + 
                  (['is_repeat'] if 'is_repeat' in top_customers.columns else [])],
    use_container_width=True,
    height=400
)

# Download button
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Data (CSV)",
    data=csv,
    file_name='customer_risk_filtered.csv',
    mime='text/csv'
)


# ===== FOOTER =====
st.markdown("---")
st.caption(
    "📊 Dashboard dibuat dengan Streamlit | "
    "Model: XGBoost + SMOTE | "
    f"Total data: {len(df):,} customers"
)