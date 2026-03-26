import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import io
from detector import detect_food_items
from nutrition import get_nutrition_info

st.set_page_config(page_title="NutriLens", page_icon="🥗", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background: #eef5ee;
    color: #1a2e1a;
}
.stApp { background: #eef5ee; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a7a3c 0%, #145c2e 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #e0f5e9 !important; }
[data-testid="stSidebar"] .stSlider * { color: #e0f5e9 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #e0f5e9 !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #145c2e 0%, #2ecc71 100%);
    border-radius: 24px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 32px rgba(26,122,60,0.18);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.hero-sub { font-size: 0.95rem; color: rgba(255,255,255,0.82); margin-top: 0.5rem; }
.hero-badge {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.32);
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.7rem;
    color: #fff;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    display: inline-block;
    margin-bottom: 0.8rem;
}
.hero-emoji { font-size: 5rem; filter: drop-shadow(0 4px 12px rgba(0,0,0,0.15)); }

/* ── Section label ── */
.slabel {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #1a7a3c;
    margin-bottom: 0.6rem;
    display: block;
}

/* ── Panel / card wrapper ── */
.panel {
    background: #ffffff;
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 16px rgba(26,122,60,0.08);
    border: 1px solid #cde8d4;
    margin-bottom: 1.2rem;
}

/* ── Metric cards ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0 1.4rem;
}
.mc {
    background: linear-gradient(145deg, #ffffff, #f0faf3);
    border: 1.5px solid #cde8d4;
    border-radius: 18px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    box-shadow: 0 3px 12px rgba(46,204,113,0.08);
    position: relative;
    overflow: hidden;
}
.mc::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 0 0 18px 18px;
}
.mc.cal::after { background: linear-gradient(90deg,#ff6b6b,#ffa07a); }
.mc.pro::after { background: linear-gradient(90deg,#2ecc71,#1abc9c); }
.mc.carb::after { background: linear-gradient(90deg,#f9ca24,#f0932b); }
.mc.fat::after  { background: linear-gradient(90deg,#a29bfe,#6c5ce7); }
.mv {
    font-size: 1.55rem;
    font-weight: 800;
    color: #1a2e1a;
    line-height: 1.1;
}
.ml {
    font-size: 0.65rem;
    color: #5a8a6a;
    margin-top: 0.28rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.08em;
}

/* ── Food tags ── */
.food-tags { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.8rem 0; }
.food-tag {
    background: linear-gradient(135deg, #e8f8ef, #d4f0e0);
    border: 1.5px solid #a8dbb8;
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    color: #1a5c30;
    font-weight: 700;
}

/* ── Bars ── */
.bar-wrap { margin: 0.55rem 0; }
.bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #3a6a3a;
    margin-bottom: 0.28rem;
    font-weight: 600;
}
.bar-bg {
    background: #d4ead9;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 999px; }

/* ── Divider ── */
.divider { border: none; border-top: 1.5px solid #d4ead9; margin: 1.2rem 0; }

/* ── Sidebar card ── */
.scard {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploadDropzone"] {
    background: #ffffff !important;
    border: 2px dashed #8ecfa0 !important;
    border-radius: 18px !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.10);
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1a7a3c, #2ecc71) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.8rem !important;
    box-shadow: 0 4px 14px rgba(46,204,113,0.3) !important;
    transition: all 0.2s !important;
}

h2, h3 { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if "daily_log" not in st.session_state:
    st.session_state.daily_log = []

# ── Sidebar ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="slabel" style="color:#a8dbb8;">📋 Daily Log</span>', unsafe_allow_html=True)
    if st.session_state.daily_log:
        total_day = sum(i["calories"] for i in st.session_state.daily_log)
        st.markdown(f"""<div class="scard" style="text-align:center;">
            <div style="font-size:0.65rem;color:#a8dbb8;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;">Total Today</div>
            <div style="font-size:2.4rem;font-weight:800;color:#ffffff;line-height:1.1;">{total_day:.0f}</div>
            <div style="color:#a8dbb8;font-size:0.75rem;font-weight:600;">kcal</div>
        </div>""", unsafe_allow_html=True)
        pct = min(total_day / 2000 * 100, 100)
        bar_color = "#ff6b6b" if pct > 90 else "#2ecc71"
        st.markdown(f"""<div class="bar-wrap">
            <div class="bar-label" style="color:#a8dbb8;"><span>Daily Goal</span>
                <span style="color:{bar_color};font-weight:700;">{pct:.0f}%</span></div>
            <div class="bar-bg" style="background:rgba(255,255,255,0.15);">
                <div class="bar-fill" style="width:{pct}%;background:{bar_color};"></div></div>
            <div style="font-size:0.68rem;color:#a8dbb8;margin-top:0.3rem;">Target: 2000 kcal</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<hr style="border-color:rgba(255,255,255,0.15);margin:0.8rem 0;">', unsafe_allow_html=True)
        for e in st.session_state.daily_log:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:0.35rem 0;
                border-bottom:1px solid rgba(255,255,255,0.1);font-size:0.8rem;font-weight:500;color:#e0f5e9;">
                <span>🍴 {e['food']}</span>
                <span style="color:#2ecc71;font-weight:700;">{e['calories']:.0f}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Log"):
            st.session_state.daily_log = []
            st.rerun()
    else:
        st.markdown("""<div style="color:#a8dbb8;font-size:0.82rem;padding:1.2rem;text-align:center;
            border:1.5px dashed rgba(255,255,255,0.2);border-radius:12px;">
            No meals logged yet.</div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="slabel" style="color:#a8dbb8;">⚙️ Settings</span>', unsafe_allow_html=True)
    daily_goal = st.slider("Daily Calorie Goal (kcal)", 1200, 4000, 2000, 50, label_visibility="visible")

# ── Hero ─────────────────────────────────────────────
st.markdown("""<div class="hero">
    <div>
        <div class="hero-badge">✦ Computer Vision · Offline AI</div>
        <p class="hero-title">NutriLens</p>
        <p class="hero-sub">Upload a meal photo for instant calorie &amp; macro breakdown.</p>
    </div>
    <div class="hero-emoji">🥗</div>
</div>""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a meal image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    w, h = image.size
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="slabel">📷 Original</span>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(f'<div style="font-size:0.7rem;color:#5a8a6a;margin-top:0.3rem;">{w}×{h}px · {uploaded_file.name}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("🔍 Analysing meal…"):
        detected_items, annotated_image = detect_food_items(image)

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="slabel">🔍 Detected Components</span>', unsafe_allow_html=True)
        st.image(annotated_image, use_container_width=True)
        if detected_items:
            tags = "".join(f'<span class="food-tag">{i.title()}</span>' for i in detected_items)
            st.markdown(f'<div class="food-tags">{tags}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if not detected_items or detected_items == ["mixed meal"]:
        st.warning("Could not detect specific items. Showing estimate for a mixed meal.")
        nutrition_data = [get_nutrition_info("mixed meal")]
    else:
        with st.spinner("Fetching nutrition data…"):
            nutrition_data = [get_nutrition_info(item) for item in detected_items]

    with col2:
        df = pd.DataFrame(nutrition_data)
        df.index = df.index + 1
        total_cal  = df["Calories (kcal)"].sum()
        total_pro  = df["Protein (g)"].sum()
        total_carb = df["Carbs (g)"].sum()
        total_fat  = df["Fat (g)"].sum()

        # Metric cards
        st.markdown(f"""<div class="metrics-row">
            <div class="mc cal">
                <div style="font-size:1.5rem;">🔥</div>
                <div class="mv">{total_cal:.0f}</div>
                <div class="ml">Calories</div>
            </div>
            <div class="mc pro">
                <div style="font-size:1.5rem;">💪</div>
                <div class="mv">{total_pro:.1f}g</div>
                <div class="ml">Protein</div>
            </div>
            <div class="mc carb">
                <div style="font-size:1.5rem;">🌾</div>
                <div class="mv">{total_carb:.1f}g</div>
                <div class="ml">Carbs</div>
            </div>
            <div class="mc fat">
                <div style="font-size:1.5rem;">🫙</div>
                <div class="mv">{total_fat:.1f}g</div>
                <div class="ml">Fat</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Macro bars
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="slabel">📊 Macro Split</span>', unsafe_allow_html=True)
        total_macros = total_pro + total_carb + total_fat or 1
        for name, val, color in [
            ("Protein", total_pro,  "#2ecc71"),
            ("Carbs",   total_carb, "#f9ca24"),
            ("Fat",     total_fat,  "#a29bfe")
        ]:
            pct = val / total_macros * 100
            st.markdown(f"""<div class="bar-wrap">
                <div class="bar-label">
                    <span>{name}</span>
                    <span style="color:{color};font-weight:700;">{val:.1f}g &nbsp;·&nbsp; {pct:.1f}%</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width:{pct}%;background:{color};"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Pie chart
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="slabel">🥧 Distribution</span>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        macro_cal = [total_pro * 4, total_carb * 4, total_fat * 9]
        wedges, texts, autotexts = ax.pie(
            macro_cal,
            labels=["Protein", "Carbs", "Fat"],
            autopct="%1.1f%%",
            colors=["#2ecc71", "#f9ca24", "#a29bfe"],
            startangle=90,
            wedgeprops=dict(edgecolor="#eef5ee", linewidth=3, width=0.6),
            pctdistance=0.75
        )
        for t in texts:    t.set_color("#1a5c30"); t.set_fontsize(10); t.set_fontweight("700")
        for t in autotexts: t.set_color("#1a2e1a"); t.set_fontsize(9)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#ffffff")
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        # Table
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="slabel">🗂️ Per-Item Breakdown</span>', unsafe_allow_html=True)
        st.dataframe(
            df.style
              .format({"Calories (kcal)": "{:.0f}", "Protein (g)": "{:.1f}", "Carbs (g)": "{:.1f}", "Fat (g)": "{:.1f}"})
              .background_gradient(subset=["Calories (kcal)"], cmap="Greens"),
            use_container_width=True,
            height=min(35 * len(df) + 40, 300)
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Goal progress
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="slabel">🎯 Goal Progress</span>', unsafe_allow_html=True)
        mp = min(total_cal / daily_goal * 100, 100)
        lp = min(sum(i["calories"] for i in st.session_state.daily_log) / daily_goal * 100, 100)
        lc = sum(i["calories"] for i in st.session_state.daily_log)
        st.markdown(f"""
        <div class="bar-wrap" style="margin-bottom:0.8rem;">
            <div class="bar-label">
                <span>This meal</span>
                <span style="color:#ff6b6b;font-weight:700;">{total_cal:.0f} / {daily_goal} kcal &nbsp;({mp:.0f}%)</span>
            </div>
            <div class="bar-bg" style="height:10px;">
                <div class="bar-fill" style="width:{mp}%;background:linear-gradient(90deg,#ff6b6b,#ffa07a);"></div>
            </div>
        </div>
        <div class="bar-wrap">
            <div class="bar-label">
                <span>Total logged</span>
                <span style="color:#2ecc71;font-weight:700;">{lc:.0f} kcal &nbsp;({lp:.0f}%)</span>
            </div>
            <div class="bar-bg" style="height:10px;">
                <div class="bar-fill" style="width:{lp}%;background:linear-gradient(90deg,#2ecc71,#1abc9c);"></div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add Meal to Daily Log"):
            for row in nutrition_data:
                st.session_state.daily_log.append({"food": row["Food Item"], "calories": row["Calories (kcal)"]})
            st.success(f"✅ Added {len(nutrition_data)} items to your log!")
            st.rerun()

else:
    st.markdown("""<div style="text-align:center;padding:4rem 2rem;
        background:linear-gradient(135deg,#ffffff,#f0faf3);
        border:2px dashed #8ecfa0;border-radius:24px;margin-top:1rem;
        box-shadow:0 4px 20px rgba(46,204,113,0.08);">
        <div style="font-size:4rem;margin-bottom:1rem;">📸</div>
        <div style="font-size:1.3rem;font-weight:800;color:#1a2e1a;margin-bottom:0.5rem;">
            Drop a meal photo above</div>
        <div style="color:#5a8a6a;font-size:0.9rem;font-weight:500;">
            Supports JPG and PNG · Works fully offline</div>
    </div>""", unsafe_allow_html=True)
