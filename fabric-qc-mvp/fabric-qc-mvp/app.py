"""FabricEye QC: AI fabric inspection demo for interview presentation.

Run:  streamlit run app.py
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import core

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "models" / "best.pt"

st.set_page_config(page_title="FabricEye QC", page_icon="🧵", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
.small-note {color: #8a8f98; font-size: 0.85rem;}
.badge {display:inline-block; padding:4px 12px; border-radius:14px; font-weight:600;}
.ok {background:#e6f4ea; color:#1e7b3a;} .warn {background:#fff4e0; color:#a35c00;} .bad {background:#fde8e8; color:#b3261e;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return core.load_model(MODEL_PATH)


@st.cache_data
def get_history():
    return core.simulate_history()


def to_bgr(uploaded) -> np.ndarray:
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


model = get_model()

# Sidebar
with st.sidebar:
    st.title("🧵 FabricEye QC")
    st.caption("AI fabric inspection demo")
    if model is not None:
        st.success("Trained YOLO model loaded")
        engine = st.radio("Detection engine", ["YOLO model", "Classical CV"], index=0)
    else:
        st.info("No trained model found in models/best.pt. Using the classical computer vision detector.")
        engine = "Classical CV"
    conf = st.slider("Confidence / sensitivity", 0.05, 0.95, 0.35 if engine == "YOLO model" else 0.5, 0.05)
    st.divider()
    st.subheader("Inspection settings")
    px_per_inch = st.number_input("Camera calibration (pixels per inch)", 10.0, 400.0, 40.0, 5.0,
                                  help="Measured once per camera by imaging a ruler.")
    width_in = st.number_input("Fabric width (inches)", 30.0, 120.0, 58.0, 1.0)
    limit = st.number_input("Acceptance limit (points per 100 sq yd)", 10.0, 80.0, 40.0, 1.0)


def run_detect(img):
    if engine == "YOLO model" and model is not None:
        return core.detect_yolo(model, img, conf)
    return core.detect_classical(img, conf)


st.title("FabricEye QC")
st.markdown("Catch fabric defects at the roll stage, grade every roll with the four point system, "
            "and turn each defect into data on machines, shifts and suppliers.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Inspect image", "📜 Roll report", "🏭 Factory analytics",
                                        "💰 Cost impact", "ℹ️ How it works"])

# Tab 1: single image
with tab1:
    c_in, c_out = st.columns([1, 2])
    with c_in:
        src = st.radio("Image source", ["Demo sample", "Upload"], horizontal=True)
        img = None
        if src == "Upload":
            up = st.file_uploader("Fabric image", type=["jpg", "jpeg", "png", "bmp"])
            if up:
                img = to_bgr(up)
        else:
            kind = st.selectbox("Sample defect", ["Hole", "Oil", "Crack", "Cutting", "none"])
            seed = st.number_input("Sample number", 0, 999, 3)
            img = core.synth_fabric(kind, int(seed))
            st.markdown('<p class="small-note">Synthetic denim texture, used only so the demo works offline. '
                        'Upload real images from the dataset for the actual demo.</p>', unsafe_allow_html=True)

    if img is not None:
        defects = run_detect(img)
        table = core.defects_table(defects, px_per_inch, img_h=img.shape[0])
        with c_out:
            a, b = st.columns(2)
            a.image(rgb(img), caption="Input", width="stretch")
            b.image(rgb(core.draw_defects(img, defects)), caption="Detected defects", width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("Defects found", len(defects))
        m2.metric("Penalty points", int(table["points"].sum()) if len(table) else 0)
        m3.metric("Engine", engine)
        if len(table):
            st.dataframe(table[["label", "confidence", "length_in", "points", "width_position_in"]],
                         width="stretch", hide_index=True)
        else:
            st.success("No defect found in this image.")

# Tab 2: roll report
with tab2:
    st.markdown("A camera captures frames while the roll runs on the inspection machine. "
                "Each frame is checked, defects are placed on a roll map, and the roll gets a grade.")
    r1, r2, r3 = st.columns(3)
    roll_id = r1.text_input("Roll ID", "HM_DN_2611_0147")
    frame_len = r2.number_input("Fabric length per frame (m)", 0.2, 5.0, 1.0, 0.1)
    mode = r3.radio("Frames", ["Simulate a demo roll", "Upload frames"], horizontal=False)

    frames = []
    if mode == "Upload frames":
        ups = st.file_uploader("Frames in roll order", type=["jpg", "jpeg", "png", "bmp"],
                               accept_multiple_files=True)
        frames = [to_bgr(u) for u in ups] if ups else []
    else:
        n_frames = st.slider("Frames in demo roll", 10, 120, 60)
        rng = np.random.default_rng(len(roll_id))
        kinds = rng.choice(["none", "Hole", "Oil", "Crack", "Cutting"], size=n_frames,
                           p=[0.72, 0.06, 0.1, 0.09, 0.03])
        frames = [core.synth_fabric(k, i, size=(320, 320)) for i, k in enumerate(kinds)]

    if frames and st.button("Run roll inspection", type="primary"):
        tables = []
        prog = st.progress(0.0)
        for i, f in enumerate(frames):
            d = run_detect(f)
            tables.append(core.defects_table(d, px_per_inch, frame_no=i + 1,
                                             frame_offset_m=i * frame_len, frame_len_m=frame_len,
                                             img_h=f.shape[0]))
            prog.progress((i + 1) / len(frames))
        prog.empty()
        all_def = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        st.session_state["roll"] = (roll_id, len(frames) * frame_len, all_def, frames)

    if "roll" in st.session_state:
        rid, length_m, all_def, frames_ = st.session_state["roll"]
        total_pts = int(all_def["points"].sum()) if len(all_def) else 0
        ppsy = core.points_per_100_sq_yd(total_pts, length_m, width_in)
        decision = core.roll_decision(ppsy, limit)
        css = "ok" if decision.startswith("Grade A") else "warn" if decision.startswith("Grade B") else "bad"

        st.markdown(f"#### Roll {rid}  <span class='badge {css}'>{decision}</span>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Length inspected", f"{length_m:.0f} m")
        k2.metric("Defects", len(all_def))
        k3.metric("Total points", total_pts)
        k4.metric("Points per 100 sq yd", f"{ppsy:.1f}", delta=f"limit {limit:.0f}", delta_color="off")

        if len(all_def):
            left, right = st.columns([3, 2])
            fig = px.scatter(all_def, x="width_position_in", y="roll_position_m", color="label",
                             size="points", hover_data=["frame", "length_in", "confidence"],
                             color_discrete_map={"Hole": "#e63c3c", "Oil": "#f0a01e",
                                                 "Crack": "#2878dc", "Cutting": "#c83cb4"},
                             title="Defect map along the roll (send this to cutting)")
            fig.update_yaxes(autorange="reversed", title="Roll position (m)")
            fig.update_xaxes(range=[0, width_in], title="Across width (inches)")
            fig.update_layout(height=460, margin=dict(t=50, b=10))
            left.plotly_chart(fig, width="stretch")
            summ = all_def.groupby("label").agg(count=("label", "size"), points=("points", "sum")).reset_index()
            right.plotly_chart(px.bar(summ, x="label", y="points", color="label", text="count",
                                      title="Points by defect type",
                                      color_discrete_map={"Hole": "#e63c3c", "Oil": "#f0a01e",
                                                          "Crack": "#2878dc", "Cutting": "#c83cb4"}
                                      ).update_layout(showlegend=False, height=460, margin=dict(t=50, b=10)),
                               width="stretch")

            worst = all_def.sort_values("points", ascending=False).head(6)
            st.markdown("**Worst frames**")
            cols = st.columns(min(6, len(worst)))
            for col, (_, row) in zip(cols, worst.iterrows()):
                f = frames_[int(row["frame"]) - 1]
                fd = [core.Defect(row["label"], row["confidence"], int(row["x1"]), int(row["y1"]),
                                  int(row["x2"]), int(row["y2"]))]
                col.image(rgb(core.draw_defects(f, fd)), caption=f"{row['roll_position_m']} m, {row['label']}",
                          width="stretch")

            st.download_button("Download roll report (CSV)", all_def.to_csv(index=False).encode(),
                               file_name=f"{rid}_defects.csv", mime="text/csv")

# Tab 3: analytics
with tab3:
    hist = get_history()
    st.warning("Simulated data for the demo. In a real pilot this table fills from every inspected roll.")
    f1, f2 = st.columns(2)
    fab = f1.multiselect("Fabric type", core.FABRICS, default=core.FABRICS)
    sh = f2.multiselect("Shift", ["Day", "Night"], default=["Day", "Night"])
    h = hist[hist["fabric"].isin(fab) & hist["shift"].isin(sh)]

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Rolls inspected", f"{len(h):,}")
    a2.metric("Meters inspected", f"{h['length_m'].sum():,.0f}")
    a3.metric("Rejection rate", f"{h['rejected'].mean() * 100:.1f}%")
    a4.metric("Avg points per 100 sq yd", f"{h['points_per_100_sq_yd'].mean():.1f}")

    by_m = h.groupby(["machine", "shift"])["points_per_100_sq_yd"].mean().reset_index()
    fig_m = px.bar(by_m, x="machine", y="points_per_100_sq_yd", color="shift", barmode="group",
                   title="Average points per 100 sq yd by machine and shift",
                   color_discrete_map={"Day": "#7fa7d9", "Night": "#1f3b73"})
    fig_m.add_hline(y=limit, line_dash="dot", annotation_text="acceptance limit")
    st.plotly_chart(fig_m, width="stretch")

    b1, b2 = st.columns(2)
    sup = h.groupby("supplier")[core.CLASSES].sum().reset_index().melt(
        id_vars="supplier", var_name="defect", value_name="count")
    b1.plotly_chart(px.bar(sup, x="supplier", y="count", color="defect", title="Defect types by yarn supplier",
                           color_discrete_map={"Hole": "#e63c3c", "Oil": "#f0a01e",
                                               "Crack": "#2878dc", "Cutting": "#c83cb4"}),
                    width="stretch")
    trend = h.groupby("date")["rejected"].mean().reset_index()
    trend["rejected"] *= 100
    b2.plotly_chart(px.line(trend, x="date", y="rejected", title="Daily roll rejection rate (%)"),
                    width="stretch")

    worst_m = by_m.sort_values("points_per_100_sq_yd", ascending=False).iloc[0]
    worst_s = h.groupby("supplier")["points_per_100_sq_yd"].mean().idxmax()
    st.info(f"**Auto insight:** {worst_m['machine']} on {worst_m['shift'].lower()} shift has the highest defect "
            f"score ({worst_m['points_per_100_sq_yd']:.1f}). {worst_s} lots carry the most yarn related cracks. "
            f"Action: schedule maintenance on {worst_m['machine']} and review {worst_s} lot quality.")

# Tab 4: cost
with tab4:
    st.markdown("Every number below is an **assumption you can change**. Ask the QC team for their real figures.")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("**Volume and defects**")
        meters = st.number_input("Fabric inspected per month (m)", 10_000, 10_000_000, 500_000, 50_000)
        dpm = st.number_input("Defects per 1,000 m", 1.0, 200.0, 10.0, 1.0)
        manual = st.slider("Manual catch rate", 0.3, 0.95, 0.65, 0.01)
        ai = st.slider("AI catch rate (target)", 0.5, 0.99, 0.85, 0.01)
    with i2:
        st.markdown("**Cost of a defect (BDT)**")
        c_fab = st.number_input("Cost if caught at fabric stage", 10, 10_000, 100, 10)
        m_cut = st.number_input("Multiplier if found at cutting", 1.0, 50.0, 3.0, 0.5)
        m_gar = st.number_input("Multiplier if found in garment or by buyer", 1.0, 500.0, 20.0, 1.0)
        share = st.slider("Share of escaped defects reaching garment stage", 0.0, 1.0, 0.1, 0.05)
    with i3:
        st.markdown("**Labour and fabric**")
        insp = st.number_input("Inspector positions moved to other work", 0, 200, 2)
        sal = st.number_input("Monthly cost per inspector (BDT)", 5_000, 100_000, 18_000, 1_000)
        fab_pct = st.number_input("Fabric saved by defect aware cutting (%)", 0.0, 5.0, 0.05, 0.05)
        price = st.number_input("Fabric price per meter (BDT)", 50, 2_000, 300, 10)
        pilot = st.number_input("Pilot cost, one inspection machine (BDT)", 50_000, 5_000_000, 800_000, 50_000)

    r = core.cost_impact(meters, dpm, manual, ai, c_fab, m_cut, m_gar, share, insp, sal, fab_pct, price)
    st.divider()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Extra defects caught / month", f"{r['extra_defects_caught']:,.0f}")
    s2.metric("Estimated monthly saving", core.bdt(r["total_monthly"]))
    s3.metric("Estimated yearly saving", core.bdt(r["total_monthly"] * 12))
    payback = pilot / r["total_monthly"] if r["total_monthly"] > 0 else float("inf")
    s4.metric("Pilot payback", f"{payback:.1f} months" if payback != float("inf") else "n/a")

    parts = pd.DataFrame({
        "source": ["Earlier detection", "Inspector time", "Fabric saved at cutting"],
        "BDT": [r["saving_late_detection"], r["saving_labour"], r["saving_fabric_cutting"]],
    })
    st.plotly_chart(px.bar(parts, x="BDT", y="source", orientation="h", text=parts["BDT"].map(core.bdt),
                           title="Where the monthly saving comes from").update_layout(height=300, yaxis_title=""),
                    width="stretch")
    st.caption("Model: extra defects caught = defects × (AI rate minus manual rate). Each one avoids the cost of "
               "being found later (cutting or garment stage) minus the cost of fixing it at fabric stage.")

# Tab 5: about
with tab5:
    st.markdown("""
### Pipeline
1. **Capture:** line scan or area camera with LED bar light over the inspection machine.
2. **Detect:** YOLO model trained on fabric defect images finds Hole, Oil, Crack and Cutting defects.
   For new fabrics with few defect samples, an anomaly model (PatchCore) learns normal fabric only.
3. **Measure:** camera calibration converts pixels to inches.
4. **Grade:** four point system gives points per defect and points per 100 square yards per roll.
5. **Act:** roll map goes to cutting, rejected rolls are held, and every defect is logged with
   machine, shift and yarn supplier for root cause analysis.

### This demo
| Part | Status |
|---|---|
| Defect detection | YOLO model trained on a public dataset (or classical CV fallback) |
| Four point grading and roll map | Working |
| Factory analytics | Simulated data |
| Cost model | Editable assumptions in BDT |

### Pilot plan
Data collection (2 to 4 weeks) → first model on Hameem fabric → shadow mode beside human
inspectors (4 weeks) → compare catch rates → expand to more machines.
""")
