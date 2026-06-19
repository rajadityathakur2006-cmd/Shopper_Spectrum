import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1976D2, #7B1FA2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-subtitle {
        color: #666;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    .divider { border-top: 2px solid #f0f0f0; margin: 1rem 0; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        padding: 0.5rem 0;
        border-bottom: 3px solid;
        margin-bottom: 1rem;
    }
    .rec-card {
        background: #f8f9ff;
        border-left: 4px solid #1976D2;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 14px;
    }
    .rec-rank {
        font-weight: 700;
        color: #1976D2;
        font-size: 16px;
    }
    .rec-score {
        background: #e3f2fd;
        color: #1565C0;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .segment-card {
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
        border-left: 6px solid;
    }
    .segment-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .segment-desc {
        font-size: 14px;
        margin: 0;
        opacity: 0.85;
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1976D2;
    }
    .metric-lbl {
        font-size: 12px;
        color: #888;
        margin-top: 2px;
    }
    .tip-box {
        background: #fff8e1;
        border-left: 4px solid #FFC107;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 13px;
        margin-top: 8px;
    }
    .stButton > button {
        width: 100%;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    try:
        kmeans  = joblib.load('models/kmeans_model.pkl')
        scaler  = joblib.load('models/scaler.pkl')
        sim_df  = joblib.load('models/similarity_matrix.pkl')
        return kmeans, scaler, sim_df, None
    except Exception as e:
        return None, None, None, str(e)

kmeans, scaler, sim_df, load_error = load_models()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Shopper Spectrum")
    st.markdown("---")
    page = st.radio(
        "Navigate to:",
        ["🏠 Home", "🎯 Product Recommendation", "👥 Customer Segmentation"],
        index=0
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses:
    - **K-Means Clustering** for customer segmentation
    - **RFM Analysis** for feature engineering
    - **Cosine Similarity** for product recommendations
    """)
    st.markdown("---")
    st.caption("Shopper Spectrum · Aditya Raj")


# ── Segment config ─────────────────────────────────────────────────────────
SEGMENT_CONFIG = {
    'High-Value': {
        'color'   : '#F9A825',
        'bg'      : '#FFFDE7',
        'icon'    : '⭐',
        'desc'    : 'Recent, frequent, and high spenders. Your most valuable customers.',
        'strategy': '🎁 Offer exclusive VIP rewards, early access to new products, and personalized thank-you messages.',
        'traits'  : ['Low Recency', 'High Frequency', 'High Monetary'],
    },
    'Regular': {
        'color'   : '#1976D2',
        'bg'      : '#E3F2FD',
        'icon'    : '🔵',
        'desc'    : 'Steady, consistent buyers with moderate spending.',
        'strategy': '🏆 Introduce loyalty programs, bundle offers, and upsell campaigns to increase spend.',
        'traits'  : ['Medium Recency', 'Medium Frequency', 'Medium Monetary'],
    },
    'Occasional': {
        'color'   : '#388E3C',
        'bg'      : '#E8F5E9',
        'icon'    : '🟢',
        'desc'    : 'Rare, infrequent buyers who purchase only sometimes.',
        'strategy': '📧 Send re-engagement emails, limited-time offers, and personalized product suggestions.',
        'traits'  : ['High Recency', 'Low Frequency', 'Low Monetary'],
    },
    'At-Risk': {
        'color'   : '#D32F2F',
        'bg'      : '#FFEBEE',
        'icon'    : '🔴',
        'desc'    : 'Customers who have not purchased in a very long time.',
        'strategy': '🚨 Launch win-back campaigns with special discounts and "we miss you" messages urgently.',
        'traits'  : ['Very High Recency', 'Very Low Frequency', 'Very Low Monetary'],
    },
}

# cluster number → segment name (based on RFM mean auto-labelling logic from notebook)
CLUSTER_TO_SEGMENT = {0: 'Regular', 1: 'Occasional', 2: 'High-Value', 3: 'At-Risk'}


# ══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <p class="main-title">🛒 Shopper Spectrum</p>
        <p class="main-subtitle">Customer Segmentation and Product Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if load_error:
        st.error(f"⚠️ Could not load models: {load_error}")
        st.info("Make sure the `models/` folder with `.pkl` files is in the same directory as `app.py`.")
    else:
        st.success("✅ All models loaded successfully!")

    st.markdown("### What this app does")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div style="background:#E3F2FD;border-radius:12px;padding:20px;height:200px">
            <h3 style="color:#1976D2;margin-top:0">🎯 Product Recommendation</h3>
            <p style="color:#333;font-size:14px">
                Enter any product name and instantly get
                <b>5 similar products</b> recommended using
                Item-Based Collaborative Filtering and Cosine Similarity.
            </p>
            <p style="color:#1976D2;font-size:13px;font-weight:600">
                → Use the sidebar to navigate there
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background:#E8F5E9;border-radius:12px;padding:20px;height:200px">
            <h3 style="color:#388E3C;margin-top:0">👥 Customer Segmentation</h3>
            <p style="color:#333;font-size:14px">
                Enter a customer's <b>Recency</b>, <b>Frequency</b>, and <b>Monetary</b>
                values and predict which of 4 segments they belong to:
                High-Value, Regular, Occasional, or At-Risk.
            </p>
            <p style="color:#388E3C;font-size:13px;font-weight:600">
                → Use the sidebar to navigate there
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Customer Segments Overview")

    cols = st.columns(4)
    for col, (seg, cfg) in zip(cols, SEGMENT_CONFIG.items()):
        with col:
            st.markdown(f"""
            <div style="background:{cfg['bg']};border-radius:10px;
                        border-top:4px solid {cfg['color']};padding:16px;text-align:center">
                <div style="font-size:2rem">{cfg['icon']}</div>
                <div style="font-weight:700;color:{cfg['color']};font-size:15px;margin:6px 0">{seg}</div>
                <div style="font-size:12px;color:#555">{cfg['desc']}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: PRODUCT RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "🎯 Product Recommendation":

    st.markdown("""
    <p class="section-header" style="color:#1976D2;border-color:#1976D2">
        🎯 Product Recommendation
    </p>
    """, unsafe_allow_html=True)
    st.markdown("Enter a product name to get **5 similar product recommendations** based on collaborative filtering.")

    if load_error or sim_df is None:
        st.error("Models not loaded. Please check the `models/` folder.")
        st.stop()

    # ── Input ──────────────────────────────────────────────────────────
    product_input = st.text_input(
        "Product Name",
        placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER",
        help="Type the product name exactly or partially. The system will find the closest match."
    )

    col_btn, col_sample = st.columns([1, 2])
    with col_btn:
        search_clicked = st.button("🔍 Get Recommendations", use_container_width=True)
    with col_sample:
        sample_clicked = st.button("🎲 Try a Sample Product", use_container_width=True)

    # ── Sample product button ──────────────────────────────────────────
    if sample_clicked:
        sample_products = [
            'WHITE HANGING HEART T-LIGHT HOLDER',
            '10 COLOUR SPACEBOY PEN',
            'JUMBO BAG RED RETROSPOT',
            'ASSORTED COLOUR BIRD ORNAMENT',
        ]
        import random
        product_input = random.choice(sample_products)
        st.info(f"🎲 Sample selected: **{product_input}**")
        search_clicked = True

    # ── Recommendation logic ───────────────────────────────────────────
    if search_clicked:
        if not product_input or not product_input.strip():
            st.warning("⚠️ Please enter a product name first.")
        else:
            query = product_input.strip().upper()

            # Exact match first, then partial
            if query in sim_df.index:
                matched = query
            else:
                matches = [p for p in sim_df.index if query in p.upper()]
                matched = matches[0] if matches else None

            if matched is None:
                st.error(f"❌ No product found matching **'{product_input}'**.")
                st.markdown("""
                <div class="tip-box">
                    💡 <b>Tips:</b> Try using fewer words (e.g. "HEART" instead of "WHITE HEART"),
                    check spelling, or use uppercase letters.
                </div>
                """, unsafe_allow_html=True)
            else:
                similar = sim_df[matched].sort_values(ascending=False).drop(matched).head(5)

                st.markdown(f"**Showing recommendations for:** `{matched}`")
                st.markdown("---")

                for i, (prod, score) in enumerate(similar.items(), 1):
                    bar_width = int(score * 100)
                    st.markdown(f"""
                    <div class="rec-card">
                        <span class="rec-rank">#{i}</span>
                        &nbsp;&nbsp;{prod}&nbsp;&nbsp;
                        <span class="rec-score">similarity: {score:.3f}</span>
                        <div style="margin-top:6px;background:#e0e0e0;border-radius:4px;height:4px">
                            <div style="width:{bar_width}%;background:#1976D2;
                                        border-radius:4px;height:4px"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("""
                <div class="tip-box">
                    💡 <b>How it works:</b> Products are compared based on which customers
                    bought them together. A higher similarity score means more customers
                    bought both products in the same transaction history.
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segmentation":

    st.markdown("""
    <p class="section-header" style="color:#7B1FA2;border-color:#7B1FA2">
        👥 Customer Segmentation
    </p>
    """, unsafe_allow_html=True)
    st.markdown("Enter a customer's **RFM values** to predict which segment they belong to.")

    if load_error or kmeans is None:
        st.error("Models not loaded. Please check the `models/` folder.")
        st.stop()

    # ── RFM explanation ────────────────────────────────────────────────
    with st.expander("ℹ️ What are RFM values?"):
        c1, c2, c3 = st.columns(3)
        c1.markdown("**🕐 Recency**\nDays since the customer last made a purchase. Lower = more recent = better.")
        c2.markdown("**🔁 Frequency**\nTotal number of unique purchases made. Higher = more loyal.")
        c3.markdown("**💰 Monetary**\nTotal amount spent by the customer in £. Higher = more valuable.")

    st.markdown("---")

    # ── Inputs ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input(
            "🕐 Recency (days)",
            min_value=0, max_value=1000, value=30,
            help="How many days ago did this customer last purchase?"
        )
    with col2:
        frequency = st.number_input(
            "🔁 Frequency (purchases)",
            min_value=0, max_value=500, value=5,
            help="How many unique transactions has this customer made?"
        )
    with col3:
        monetary = st.number_input(
            "💰 Monetary (£ spent)",
            min_value=0.0, max_value=100000.0, value=500.0, step=50.0,
            help="What is the total amount this customer has spent?"
        )

    st.markdown("")

    # ── Quick examples ─────────────────────────────────────────────────
    st.markdown("**Or try a quick example:**")
    ex1, ex2, ex3, ex4 = st.columns(4)
    examples = {
        '⭐ High-Value Example'  : (10,  25, 5000.0),
        '🔵 Regular Example'    : (60,   5,  500.0),
        '🟢 Occasional Example' : (150,  2,  100.0),
        '🔴 At-Risk Example'    : (350,  1,   50.0),
    }
    chosen_example = None
    for col, (label, vals) in zip([ex1, ex2, ex3, ex4], examples.items()):
        if col.button(label, use_container_width=True):
            chosen_example = vals

    if chosen_example:
        recency, frequency, monetary = chosen_example
        st.rerun()

    st.markdown("")
    predict_clicked = st.button("🔮 Predict Segment", use_container_width=False)

    # ── Prediction ─────────────────────────────────────────────────────
    if predict_clicked:
        input_arr    = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_arr)
        cluster_num  = int(kmeans.predict(input_scaled)[0])
        segment      = CLUSTER_TO_SEGMENT.get(cluster_num, f'Cluster {cluster_num}')
        cfg          = SEGMENT_CONFIG[segment]

        # Segment result card
        st.markdown(f"""
        <div class="segment-card"
             style="background:{cfg['bg']};border-color:{cfg['color']}">
            <p class="segment-title" style="color:{cfg['color']}">
                {cfg['icon']} {segment}
            </p>
            <p class="segment-desc" style="color:#333">{cfg['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{recency}d</div>
            <div class="metric-lbl">Recency</div>
        </div>""", unsafe_allow_html=True)
        m2.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{frequency}</div>
            <div class="metric-lbl">Frequency</div>
        </div>""", unsafe_allow_html=True)
        m3.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">£{monetary:,.0f}</div>
            <div class="metric-lbl">Monetary</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Traits
        t_cols = st.columns(len(cfg['traits']))
        for col, trait in zip(t_cols, cfg['traits']):
            col.markdown(f"""
            <div style="background:{cfg['color']}22;border-radius:20px;
                        padding:6px 12px;text-align:center;
                        color:{cfg['color']};font-size:13px;font-weight:600">
                {trait}
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Strategy tip
        st.markdown(f"""
        <div class="tip-box" style="background:{cfg['bg']};border-color:{cfg['color']}">
            <b>Recommended Strategy:</b><br>{cfg['strategy']}
        </div>
        """, unsafe_allow_html=True)

        # All segments reference
        st.markdown("---")
        st.markdown("**All 4 Segments at a Glance:**")
        cols = st.columns(4)
        for col, (seg, c) in zip(cols, SEGMENT_CONFIG.items()):
            border = "3px" if seg == segment else "1px"
            opacity = "1" if seg == segment else "0.5"
            col.markdown(f"""
            <div style="background:{c['bg']};border:{border} solid {c['color']};
                        border-radius:8px;padding:10px;text-align:center;
                        opacity:{opacity}">
                <div style="font-size:1.4rem">{c['icon']}</div>
                <div style="font-weight:700;color:{c['color']};font-size:13px">{seg}</div>
            </div>""", unsafe_allow_html=True)
