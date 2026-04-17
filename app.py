import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-size: 14px;
        font-weight: 500;
        color: #374151;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Badges */
    .badge-green { background:#dcfce7; color:#166534; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-red   { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-blue  { background:#dbeafe; color:#1e40af; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-amber { background:#fef9c3; color:#92400e; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }

    /* Page title */
    .page-title {
        font-size: 22px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 6px;
    }
    .page-sub {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 24px;
    }

    /* Result box */
    .result-normal {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 10px;
        padding: 14px 18px;
        color: #166534;
        font-size: 16px;
        font-weight: 600;
    }
    .result-attack {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 10px;
        padding: 14px 18px;
        color: #991b1b;
        font-size: 16px;
        font-weight: 600;
    }
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 12px 16px;
        color: #1e40af;
        font-size: 13px;
        margin-bottom: 12px;
    }
    .warning-box {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 10px;
        padding: 12px 16px;
        color: #92400e;
        font-size: 13px;
        margin-bottom: 12px;
    }
    div[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { font-size: 13px !important; color: #64748b !important; }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }
    .stButton > button[kind="primary"] {
        background-color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_PATH   = "model.pkl"
SCALER_PATH  = "scaler.pkl"
ENCODER_PATH = "encoders.pkl"
META_PATH    = "model_meta.pkl"

CAT_COLS     = ['proto', 'service', 'state']
DROP_COLS    = ['id', 'attack_cat']
TARGET_COL   = 'label'

# ─────────────────────────────────────────────
# Helper: preprocessing
# ─────────────────────────────────────────────
def preprocess(df, encoders=None, scaler=None, fit=True):
    df = df.copy()

    # Drop irrelevant cols
    for c in DROP_COLS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Separate target
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].values
        df.drop(columns=[TARGET_COL], inplace=True)
    else:
        y = None

    # Fix ambiguous columns: try converting everything possible to numeric first
    for col in df.columns:
        if df[col].dtype == 'object' and col not in CAT_COLS:
            converted = pd.to_numeric(df[col], errors='coerce')
            # If >50% of values converted successfully, treat as numeric
            if converted.notna().sum() > len(df) * 0.5:
                df[col] = converted

    # Fill missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown', inplace=True)

    # Encode categoricals
    present_cats = [c for c in CAT_COLS if c in df.columns]
    if fit:
        encoders = {}
        for col in present_cats:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in present_cats:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

    feature_names = df.columns.tolist()

    # Scale
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(df)
    else:
        X = scaler.transform(df)

    return X, y, encoders, scaler, feature_names

# ─────────────────────────────────────────────
# Helper: train & save
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train(df_hash, df):
    # Check if saved model exists and is fresh
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, META_PATH]):
        meta = joblib.load(META_PATH)
        if meta.get('df_hash') == df_hash:
            model    = joblib.load(MODEL_PATH)
            scaler   = joblib.load(SCALER_PATH)
            encoders = joblib.load(ENCODER_PATH)
            return model, scaler, encoders, meta, False  # False = not retrained

    # Preprocess
    X, y, encoders, scaler, feat_names = preprocess(df, fit=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, zero_division=0)
    rec    = recall_score(y_test, y_pred, zero_division=0)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    cm     = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred,
                                   target_names=['Normal', 'Attack'],
                                   output_dict=True)

    meta = {
        'df_hash':      df_hash,
        'accuracy':     acc,
        'precision':    prec,
        'recall':       rec,
        'f1':           f1,
        'cm':           cm,
        'report':       report,
        'feat_names':   feat_names,
        'n_samples':    len(df),
        'n_features':   len(feat_names),
        'n_train':      len(X_train),
        'n_test':       len(X_test),
    }

    # Save
    joblib.dump(model,    MODEL_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(meta,     META_PATH)

    return model, scaler, encoders, meta, True  # True = freshly trained

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ ML IDS Dashboard")
    st.markdown("<small style='color:#64748b'>v1.0 · UNSW-NB15 · Random Forest</small>", unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", [
        "📊 Dashboard",
        "🔍 Data Explorer",
        "⚙️ Model Training",
        "📈 Visualizations",
        "🔮 Predictions",
        "📋 Model Info"
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Upload Dataset**")
    uploaded = st.file_uploader("Choose CSV", type="csv", label_visibility="collapsed")
    st.caption("UNSW-NB15 training-set.csv")

    if uploaded:
        st.success(f"✅ {uploaded.name} loaded")

# ─────────────────────────────────────────────
# Load data — works with or without CSV upload
# If pkl files exist, load directly. CSV only
# needed for first-time training.
# ─────────────────────────────────────────────
pkl_ready = all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, META_PATH])

if uploaded is None and not pkl_ready:
    # No CSV, no saved model — show welcome screen
    st.markdown('<div class="page-title">🛡️ AI-Powered Intrusion Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload your UNSW-NB15 dataset from the sidebar to begin.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
        <h4>📥 Getting Started</h4>
        <ol style="font-size:14px; color:#374151; line-height:2">
        <li>Download UNSW-NB15 from <b>research.unsw.edu.au</b></li>
        <li>Upload <code>UNSW_NB15_training-set.csv</code> in the sidebar</li>
        <li>The model trains automatically (once)</li>
        <li>Explore all pages — predictions, charts, metrics</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
        <h4>🤖 About this Tool</h4>
        <p style="font-size:14px; color:#374151; line-height:1.8">
        This dashboard trains a <b>Random Forest</b> classifier on network traffic data
        to automatically detect intrusions. Built for <b>CLO 4 — Information Security</b>.<br><br>
        The model is <b>cached after first training</b> — no CSV needed on future visits.
        </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# Train or load model
# ─────────────────────────────────────────────
if uploaded is not None:
    # CSV uploaded — train or retrain
    df_raw = pd.read_csv(uploaded)
    df_hash = str(len(df_raw)) + str(df_raw.columns.tolist())
    try:
        with st.spinner("🔄 Training model... please wait (~30-60 seconds on first run)"):
            model, scaler, encoders, meta, was_trained = load_or_train(df_hash, df_raw)
    except Exception as e:
        st.error(f"Training failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()
    if was_trained:
        st.toast("✅ Model trained and saved successfully!", icon="🎉")
    else:
        st.toast("⚡ Loaded saved model instantly!", icon="💾")

else:
    # No CSV uploaded but pkl files exist — load directly, no CSV needed
    model      = joblib.load(MODEL_PATH)
    scaler     = joblib.load(SCALER_PATH)
    encoders   = joblib.load(ENCODER_PATH)
    meta       = joblib.load(META_PATH)
    df_raw     = None   # no raw data available without CSV
    was_trained = False
    with st.sidebar:
        st.info("⚡ Model loaded from disk. Upload CSV to retrain or explore raw data.")

feat_names = meta['feat_names']

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.markdown('<div class="page-title">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Overview of model performance and dataset statistics.</div>', unsafe_allow_html=True)

    # Model status banner
    if was_trained:
        st.markdown('<div class="info-box">🆕 Model was freshly trained on this dataset and saved to disk. Future loads will be instant.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">⚡ Loaded pre-trained model from disk — no retraining needed.</div>', unsafe_allow_html=True)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Accuracy",  f"{meta['accuracy']*100:.2f}%")
    c2.metric("🔍 Precision", f"{meta['precision']*100:.2f}%")
    c3.metric("📡 Recall",    f"{meta['recall']*100:.2f}%",   help="Most critical for IDS")
    c4.metric("⚖️ F1 Score",  f"{meta['f1']*100:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Summary")
        info_data = {
            "Property": ["Total Records", "Train Samples", "Test Samples", "Features", "Normal Traffic", "Attack Traffic"],
            "Value": [
                f"{meta['n_samples']:,}",
                f"{meta['n_train']:,}",
                f"{meta['n_test']:,}",
                str(meta['n_features']),
                f"{(df_raw['label']==0).sum():,}" if df_raw is not None and 'label' in df_raw.columns else str(meta['n_samples'] - meta['n_test']),
                f"{(df_raw['label']==1).sum():,}" if df_raw is not None and 'label' in df_raw.columns else "See model info",
            ]
        }
        st.dataframe(pd.DataFrame(info_data), hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Class Distribution")
        if df_raw is not None and 'label' in df_raw.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            counts = df_raw['label'].value_counts()
            bars = ax.bar(['Normal (0)', 'Attack (1)'], counts.values,
                          color=['#22c55e', '#ef4444'], edgecolor='white', linewidth=1.5, width=0.5)
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                        f'{val:,}', ha='center', fontsize=10, fontweight='bold', color='#1e293b')
            ax.set_ylabel('Count', fontsize=10, color='#64748b')
            ax.set_facecolor('#f8fafc')
            ax.spines[['top','right']].set_visible(False)
            fig.patch.set_facecolor('white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Feature importance preview
    st.subheader("Top 10 Feature Importances")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feat_df)))
    bars = ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors, edgecolor='white')
    for bar, val in zip(bars, feat_df['Importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, color='#374151')
    ax.set_xlabel('Gini Importance', fontsize=10, color='#64748b')
    ax.set_facecolor('#f8fafc')
    ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Data Explorer":
    st.markdown('<div class="page-title">🔍 Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Explore and understand your dataset before modelling.</div>', unsafe_allow_html=True)

    if df_raw is None:
        st.info("📂 Upload the CSV from the sidebar to explore raw data. Model metrics are still available on other pages.")
        st.stop()

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",     f"{len(df_raw):,}")
    c2.metric("Columns",  str(df_raw.shape[1]))
    c3.metric("Missing",  str(df_raw.isnull().sum().sum()))
    c4.metric("Classes",  str(df_raw['label'].nunique()) if 'label' in df_raw.columns else "N/A")

    st.subheader("Data Preview")
    n_rows = st.slider("Rows to display", 5, 50, 10)
    st.dataframe(df_raw.head(n_rows), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Statistics")
        st.dataframe(df_raw.describe().round(3), use_container_width=True)

    with col2:
        st.subheader("Missing Values")
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            st.success("✅ No missing values found!")
        else:
            st.dataframe(missing.rename("Missing Count"), use_container_width=True)

        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': df_raw.dtypes.index,
            'Type': df_raw.dtypes.values.astype(str)
        })
        st.dataframe(dtype_df, hide_index=True, use_container_width=True)

    # Attack category breakdown
    if 'attack_cat' in df_raw.columns:
        st.subheader("Attack Category Distribution")
        att_counts = df_raw['attack_cat'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 4))
        colors_att = plt.cm.Reds(np.linspace(0.4, 0.9, len(att_counts)))
        ax.barh(att_counts.index, att_counts.values, color=colors_att, edgecolor='white')
        for i, (val, bar) in enumerate(zip(att_counts.values, ax.patches)):
            ax.text(val + 50, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontsize=9)
        ax.set_xlabel('Count', fontsize=10, color='#64748b')
        ax.set_facecolor('#f8fafc')
        ax.spines[['top','right']].set_visible(False)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap (Top 12 numeric features)")
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in num_cols:
        corr_order = df_raw[num_cols].corr()['label'].abs().sort_values(ascending=False).head(12).index.tolist()
    else:
        corr_order = num_cols[:12]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_raw[corr_order].corr(), annot=True, fmt='.2f', cmap='RdYlGn',
                linewidths=0.5, square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=12, color='#1e293b')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚙️ Model Training":
    st.markdown('<div class="page-title">⚙️ Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Review training configuration and retrain if needed.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Current Configuration")
        config_data = {
            "Parameter": ["Algorithm", "n_estimators", "max_depth", "min_samples_split",
                          "Train/Test Split", "Random State", "Scaling", "Encoding"],
            "Value":     ["Random Forest", "100", "None (unlimited)", "5",
                          "80% / 20%", "42", "StandardScaler", "LabelEncoder"]
        }
        st.dataframe(pd.DataFrame(config_data), hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        💡 <b>Why Random Forest?</b><br>
        Ensemble of 100 decision trees — reduces overfitting, handles complex 
        network feature relationships, and provides feature importance scores 
        for security analyst explainability.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Model Status")
        if os.path.exists(MODEL_PATH):
            st.success("✅ Saved model found on disk")
            size_kb = os.path.getsize(MODEL_PATH) / 1024
            st.info(f"📁 model.pkl — {size_kb:.1f} KB")
        else:
            st.warning("⚠️ No saved model — will train on upload")

        st.subheader("Training Results")
        c1, c2 = st.columns(2)
        c1.metric("Training samples", f"{meta['n_train']:,}")
        c2.metric("Test samples",     f"{meta['n_test']:,}")
        c1.metric("Features used",    str(meta['n_features']))
        c2.metric("Model accuracy",   f"{meta['accuracy']*100:.2f}%")

    st.divider()
    st.subheader("Force Retrain")
    st.markdown("""
    <div class="warning-box">
    ⚠️ Retraining will delete the saved model and train from scratch. This takes ~30-60 seconds.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Delete saved model & retrain on next load", type="secondary"):
        for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, META_PATH]:
            if os.path.exists(p):
                os.remove(p)
        st.cache_resource.clear()
        st.success("✅ Saved model deleted. Refresh the page to retrain.")

    # Classification report
    st.divider()
    st.subheader("Full Classification Report")
    report = meta['report']
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Visualizations":
    st.markdown('<div class="page-title">📈 Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Charts and graphs for security analysis and reporting.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Confusion matrix
    with col1:
        st.subheader("Confusion Matrix")
        cm = np.array(meta['cm'])
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ['Normal', 'Attack']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    linewidths=2, linecolor='white', ax=ax,
                    annot_kws={'size': 16, 'weight': 'bold'})
        ax.set_xlabel('Predicted', fontsize=11, color='#374151')
        ax.set_ylabel('Actual', fontsize=11, color='#374151')
        ax.set_title('Confusion Matrix — Random Forest', fontsize=12, color='#1e293b', pad=12)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Breakdown
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
        c1i, c2i = st.columns(2)
        c1i.metric("✅ True Positives",  f"{tp:,}", help="Attacks correctly detected")
        c2i.metric("✅ True Negatives",  f"{tn:,}", help="Normal traffic correctly allowed")
        c1i.metric("⚠️ False Positives", f"{fp:,}", help="Normal flagged as attack")
        c2i.metric("🚨 False Negatives", f"{fn:,}", help="DANGEROUS: attacks missed")

    # Feature importance
    with col2:
        st.subheader("Feature Importance (Top 15)")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(5, 5))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(feat_df)))
        ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors, edgecolor='white')
        ax.set_xlabel('Gini Importance', fontsize=10, color='#64748b')
        ax.set_title('Top 15 Features', fontsize=12, color='#1e293b')
        ax.set_facecolor('#f8fafc')
        ax.spines[['top','right']].set_visible(False)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Metrics bar chart
    st.subheader("Performance Metrics Comparison")
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value':  [meta['accuracy'], meta['precision'], meta['recall'], meta['f1']]
    })
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bar_colors = ['#1d4ed8', '#7c3aed', '#059669', '#d97706']
    bars = ax.bar(metrics_df['Metric'], metrics_df['Value'],
                  color=bar_colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, metrics_df['Value']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val*100:.2f}%', ha='center', fontsize=11, fontweight='bold', color='#1e293b')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=10, color='#64748b')
    ax.set_facecolor('#f8fafc')
    ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Per-class metrics
    st.subheader("Per-Class Metrics")
    report = meta['report']
    per_class = pd.DataFrame({
        'Class':     ['Normal (0)', 'Attack (1)'],
        'Precision': [report.get('Normal',{}).get('precision', 0),
                      report.get('Attack',{}).get('precision', 0)],
        'Recall':    [report.get('Normal',{}).get('recall', 0),
                      report.get('Attack',{}).get('recall', 0)],
        'F1':        [report.get('Normal',{}).get('f1-score', 0),
                      report.get('Attack',{}).get('f1-score', 0)],
        'Support':   [int(report.get('Normal',{}).get('support', 0)),
                      int(report.get('Attack',{}).get('support', 0))],
    }).round(4)
    st.dataframe(per_class, hide_index=True, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    📌 <b>Security note:</b> High Recall for Attack class means the model catches most real attacks. 
    False Negatives (missed attacks) are the most dangerous outcome for an IDS — each represents a 
    potential undetected breach.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Predictions":
    st.markdown('<div class="page-title">🔮 Live Predictions</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Enter network flow features to classify traffic as Normal or Attack in real time.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["✏️ Manual Input", "📂 Batch CSV Upload"])

    # ── Manual Input ──────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Enter Network Flow Features")

        # Build input fields dynamically from feature names
        input_vals = {}
        n_cols = 4
        rows = [feat_names[i:i+n_cols] for i in range(0, len(feat_names), n_cols)]

        for row in rows:
            cols = st.columns(len(row))
            for col, feat in zip(cols, row):
                with col:
                    # Use sensible defaults
                    defaults = {
                        'dur': 0.5, 'sbytes': 1000, 'dbytes': 800,
                        'rate': 10.0, 'sttl': 64, 'dttl': 64,
                        'sload': 1000.0, 'dload': 800.0, 'sloss': 0,
                        'dloss': 0, 'spkts': 5, 'dpkts': 4,
                        'proto': 6, 'service': 0, 'state': 1,
                    }
                    default_val = defaults.get(feat, 0.0)
                    input_vals[feat] = st.number_input(feat, value=float(default_val), format="%.4f", key=f"inp_{feat}")

        st.markdown("---")
        pred_col, info_col = st.columns([1, 2])

        with pred_col:
            if st.button("🔮 Classify Traffic", type="primary", use_container_width=True):
                # Build input array in correct feature order
                input_array = np.array([[input_vals[f] for f in feat_names]])
                input_scaled = scaler.transform(input_array)

                prediction = model.predict(input_scaled)[0]
                probabilities = model.predict_proba(input_scaled)[0]
                confidence = probabilities[prediction] * 100

                st.session_state['last_pred'] = prediction
                st.session_state['last_conf'] = confidence
                st.session_state['last_proba'] = probabilities

        with info_col:
            if 'last_pred' in st.session_state:
                pred = st.session_state['last_pred']
                conf = st.session_state['last_conf']
                proba = st.session_state['last_proba']

                if pred == 0:
                    st.markdown(f'<div class="result-normal">✅ NORMAL TRAFFIC &nbsp;|&nbsp; Confidence: {conf:.1f}%</div>', unsafe_allow_html=True)
                    st.success("This network flow appears to be benign.")
                else:
                    st.markdown(f'<div class="result-attack">🚨 ATTACK DETECTED &nbsp;|&nbsp; Confidence: {conf:.1f}%</div>', unsafe_allow_html=True)
                    st.error("Malicious traffic detected! Immediate investigation recommended.")

                st.markdown("**Probability breakdown:**")
                p_col1, p_col2 = st.columns(2)
                p_col1.metric("Normal probability",  f"{proba[0]*100:.2f}%")
                p_col2.metric("Attack probability",  f"{proba[1]*100:.2f}%")

                # Visual probability bar
                fig, ax = plt.subplots(figsize=(5, 1.2))
                ax.barh([''], [proba[0]], color='#22c55e', label='Normal')
                ax.barh([''], [proba[1]], left=[proba[0]], color='#ef4444', label='Attack')
                ax.set_xlim(0, 1)
                ax.set_facecolor('#f8fafc')
                ax.spines[['top','right','left','bottom']].set_visible(False)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0%','25%','50%','75%','100%'], fontsize=9)
                ax.legend(loc='lower right', fontsize=8)
                fig.patch.set_facecolor('white')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # ── Batch Upload ──────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Batch Prediction from CSV")
        st.markdown("""
        <div class="info-box">
        Upload a CSV with the same feature columns as the training data (no label column needed).
        The model will classify each row and show results.
        </div>
        """, unsafe_allow_html=True)

        batch_file = st.file_uploader("Upload prediction CSV", type="csv", key="batch_pred")

        if batch_file:
            batch_df = pd.read_csv(batch_file)
            st.write(f"Loaded **{len(batch_df):,}** rows for prediction")
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("🚀 Run Batch Predictions", type="primary"):
                with st.spinner("Classifying..."):
                    # Remove label if present
                    batch_clean = batch_df.copy()
                    for c in DROP_COLS + [TARGET_COL]:
                        if c in batch_clean.columns:
                            batch_clean.drop(columns=[c], inplace=True)

                    # Fill missing
                    for col in batch_clean.columns:
                        if batch_clean[col].dtype == 'object':
                            batch_clean[col].fillna('unknown', inplace=True)
                        else:
                            batch_clean[col].fillna(0, inplace=True)

                    # Encode
                    for col in [c for c in CAT_COLS if c in batch_clean.columns]:
                        if col in encoders:
                            le = encoders[col]
                            batch_clean[col] = batch_clean[col].astype(str).apply(
                                lambda x: le.transform([x])[0] if x in le.classes_ else 0
                            )

                    # Keep only trained features
                    present = [f for f in feat_names if f in batch_clean.columns]
                    X_batch = batch_clean[present].values
                    X_batch_scaled = scaler.transform(X_batch)

                    preds = model.predict(X_batch_scaled)
                    probas = model.predict_proba(X_batch_scaled)

                    results_df = batch_df.copy()
                    results_df['Prediction'] = ['Attack' if p == 1 else 'Normal' for p in preds]
                    results_df['Confidence'] = [f"{probas[i][p]*100:.1f}%" for i, p in enumerate(preds)]

                    st.success(f"✅ Classified {len(preds):,} records")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Normal", f"{(preds==0).sum():,}")
                    c2.metric("Attack", f"{(preds==1).sum():,}")
                    c3.metric("Attack rate", f"{(preds==1).mean()*100:.1f}%")

                    st.dataframe(results_df, use_container_width=True)

                    # Download
                    csv_out = results_df.to_csv(index=False).encode()
                    st.download_button("📥 Download Results CSV", csv_out,
                                       "batch_predictions.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Model Info":
    st.markdown('<div class="page-title">📋 Model Info & Security Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Detailed breakdown for the CISO report and viva presentation.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Pipeline")
        steps = [
            ("1️⃣ Data Ingestion",    "Load UNSW-NB15 CSV into pandas DataFrame"),
            ("2️⃣ Column Drop",       "Remove 'id' and 'attack_cat' (irrelevant/leakage)"),
            ("3️⃣ Missing Values",    "Median fill (numeric) / Mode fill (categorical)"),
            ("4️⃣ Label Encoding",    "LabelEncoder on proto, service, state columns"),
            ("5️⃣ Normalisation",     "StandardScaler → mean=0, std=1 for all features"),
            ("6️⃣ Train/Test Split",  "80/20 stratified split (random_state=42)"),
            ("7️⃣ RF Training",       "RandomForestClassifier (100 trees, n_jobs=-1)"),
            ("8️⃣ Evaluation",        "Accuracy, Precision, Recall, F1, Confusion Matrix"),
            ("9️⃣ Save to Disk",      "joblib.dump() → model.pkl, scaler.pkl, encoders.pkl"),
        ]
        for step, desc in steps:
            st.markdown(f"**{step}** — {desc}")

    with col2:
        st.subheader("Security Analysis")
        st.markdown("""
        **Why Recall is the most critical metric for IDS:**
        
        Recall = TP / (TP + FN)
        
        A False Negative (FN) means a real attack was NOT detected. 
        Each FN could represent:
        - An undetected DDoS flood draining server resources
        - A Backdoor connection giving attackers persistent access  
        - Shellcode execution enabling remote code execution
        - A Reconnaissance scan mapping the internal network
        
        **A low Recall is a critical security flaw** — it means the IDS 
        is blind to real threats, giving defenders false confidence.
        """)

        st.subheader("Proposed Future Enhancements")
        enhancements = [
            "LSTM for temporal packet sequence modelling",
            "SHAP explainability for per-alert analyst justification",
            "Online learning to handle concept drift",
            "Multi-class classification (identify specific attack types)",
            "Integration with SIEM (Splunk/ELK) for automated alerting",
            "Feature engineering: bytes_per_packet, asymmetry ratios",
        ]
        for e in enhancements:
            st.markdown(f"- {e}")

    st.divider()
    st.subheader("All Features Used")
    feat_imp_df = pd.DataFrame({
        'Feature': feat_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    feat_imp_df.index += 1
    feat_imp_df['Importance'] = feat_imp_df['Importance'].round(6)
    st.dataframe(feat_imp_df, use_container_width=True)

    # Download model meta as report
    st.divider()
    st.subheader("Export")
    report_txt = f"""
ML IDS MODEL REPORT
===================
Accuracy:   {meta['accuracy']*100:.2f}%
Precision:  {meta['precision']*100:.2f}%
Recall:     {meta['recall']*100:.2f}%
F1 Score:   {meta['f1']*100:.2f}%

Dataset:    UNSW-NB15
Samples:    {meta['n_samples']:,}
Features:   {meta['n_features']}
Train:      {meta['n_train']:,}
Test:       {meta['n_test']:,}

Algorithm:  Random Forest
Trees:      100
"""
    st.download_button("📥 Download Model Report (.txt)", report_txt,
                       "model_report.txt", "text/plain")
