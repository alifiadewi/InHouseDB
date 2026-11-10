"""
File naming convention (one file per unit):
    <Group>_<DD-MMM-YYYY>_<SampleNumber>[_<Annotation>].xlsx

Run
    streamlit run app.py
"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

VALUE_COLUMNS = ["AnglePTP", "RearHubPTP"]
X_TICK_STEP   = 5
X_MINOR_STEP  = 1
SUPPORTED_EXT = {".xlsx", ".xls", ".xlsm", ".csv"}
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

# ========================================================== Data Processing
def parse_filename(stem: str):
    m = re.match(
        r"^(?P<group>.+?)_(?P<date>\d{1,2}-[A-Za-z]{3}-\d{4})_"
        r"(?P<sample>[^_]+)(?:_(?P<annotation>.+))?$", stem)
    if not m: return None
    try:
        d, mon, y = m.group("date").split("-")
        month_num = MONTHS[mon.lower()[:3]]
        dt = datetime(int(y), month_num, int(d))
    except Exception:
        return None
    return dict(group=m.group("group"), date=dt, sample=m.group("sample"), 
                annotation=m.group("annotation") or "")

def clean_str(s):
    return str(s).lower().replace(" ", "").replace("_", "")

def find_frequency_column(cols):
    for c in cols:
        lc = clean_str(c)
        if "frequency" in lc and not any(x in lc for x in ("start", "end", "interval", "duty")):
            return c
    return None

def find_value_column(cols, target):
    t = clean_str(target)
    for c in cols:
        if t in clean_str(c):
            return c
    return None

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def read_measurement(source):
    try:
        suffix = Path(source if isinstance(source, (str, Path)) else source.name).suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(source, sep=None, engine='python')
        else:
            df = pd.read_excel(source)
    except Exception as e:
        return {"error": f"Cannot read file: {e}"}

    cols = list(df.columns)
    fcol = find_frequency_column(cols)
    if fcol is None:
        return {"error": f"No frequency column found. Available: {cols}"}

    freq = to_num(df[fcol])
    curves = {}
    for name in VALUE_COLUMNS:
        vcol = find_value_column(cols, name)
        if vcol is None:
            return {"error": f"Column '{name}' not found. Available: {cols}"}
        val = to_num(df[vcol])
        mask = freq.notna() & val.notna()
        f = freq[mask].to_numpy(dtype=float)
        v = val[mask].to_numpy(dtype=float)
        order = np.argsort(f)
        curves[name] = (f[order], v[order])

    return {"curves": curves}

def load_units(items):
    # Accepts uploaded files directly from the web browser
    files = [f for f in items if not f.name.startswith("~$")]

    units, skipped = [], []
    for f in sorted(files, key=lambda p: Path(p.name).name):
        stem = Path(f.name).stem
        meta = parse_filename(stem)
        if meta is None:
            skipped.append(f"{f.name}: Bad filename format")
            continue
        data = read_measurement(f)
        if "error" in data:
            skipped.append(f"{f.name}: {data['error']}")
            continue
        meta["file"] = f.name
        meta["curves"] = data["curves"]
        units.append(meta)

    return units, f"Loaded {len(units)} units.", skipped

def make_colors(n):
    if n == 0: return []
    scale = pcolors.sequential.Turbo
    ts = [0.5] if n == 1 else np.linspace(0.0, 1.0, n).tolist()
    return pcolors.sample_colorscale(scale, ts)

# ========================================================== Plotting
def build_figure(units, shown_idx, colors, layout, legend_on, total_filtered):
    stacked = layout == "Stacked (top / bottom)"
    rows, cols = (2, 1) if stacked else (1, 2)
    
    n_shown = len(shown_idx)
    title1 = f"AnglePTP – {n_shown} of {total_filtered} shown"
    title2 = f"RearHubPTP – {n_shown} of {total_filtered} shown"
    
    fig = make_subplots(
        rows=rows, cols=cols, 
        subplot_titles=[title1, title2],
        vertical_spacing=0.12, 
        horizontal_spacing=0.08
    )

    for v_i, name in enumerate(VALUE_COLUMNS):
        r = v_i + 1 if stacked else 1
        c = 1 if stacked else v_i + 1
        lo, hi = np.inf, -np.inf

        for i in shown_idx:
            u = units[i]
            if name not in u["curves"]: continue
            f, v = u["curves"][name]
            label = u["sample"] + (f" ({u['annotation']})" if u["annotation"] else "")
            
            fig.add_trace(
                go.Scatter(
                    x=f, y=v, mode="lines", 
                    line=dict(color=colors[i], width=1.2),
                    name=label, showlegend=legend_on,
                    hovertemplate=f"<b>{label}</b><br>Freq: %{{x:.2f}} Hz<br>{name}: %{{y:.3f}}<extra></extra>"
                ),
                row=r, col=c
            )
            lo, hi = min(lo, float(f.min())), max(hi, float(f.max()))

        xname = "xaxis" if v_i == 0 else f"xaxis{v_i + 1}"
        yname = "yaxis" if v_i == 0 else f"yaxis{v_i + 1}"
        
        xcfg = dict(
            title="Frequency (Hz)", tickangle=90, 
            showgrid=True, gridcolor='#e5e5e5', zeroline=False
        )
        if np.isfinite(lo) and hi > lo:
            xcfg["range"] = [lo, hi]
            xcfg["dtick"] = X_TICK_STEP
            xcfg["minor"] = dict(ticks="outside", dtick=X_MINOR_STEP, showgrid=True, gridcolor='#f0f0f0')
        
        ycfg = dict(title=name, showgrid=True, gridcolor='#e5e5e5', zeroline=False)
        
        fig.update_layout(**{xname: xcfg})
        fig.update_layout(**{yname: ycfg})

    height = 820 if stacked else 460
    fig.update_layout(
        height=height, 
        margin=dict(l=60, r=160 if legend_on else 30, t=40, b=60),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=legend_on,
        legend=dict(orientation="v", x=1.02, xanchor="left", font=dict(size=10))
    )
    return fig

# ========================================================== UI & Layout
st.set_page_config(page_title="PTP Measurement Dashboard", layout="wide")

# --- THIS IS THE ST.MARKDOWN CSS YOU WERE MISSING ---
st.markdown("""
<style>
    /* Hide top header, menu, and footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Page padding */
    .block-container { padding-top: 2.5rem !important; padding-bottom: 1rem !important; max-width: 98% !important; }
    .stButton>button { width: 100%; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }

    /* 1. REMOVE THE OUTER GREY BOX */
    [data-testid="stFileUploadDropzone"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        min-height: auto !important;
    }

    /* 2. REMOVE THE PRINTED TEXT */
    /* Hide "200MB per file" and the uploaded file list */
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] ul {
        display: none !important;
    }
    /* Hide the cloud icon and "Drag and drop" text */
    [data-testid="stFileUploadDropzone"] > div:first-child {
        display: none !important;
    }

    /* 3. RESIZE AND STYLE THE UPLOAD BUTTON */
    [data-testid="stFileUploadDropzone"] button {
        width: 100% !important;
        min-height: 38px !important;
        border: 1px solid rgba(49, 51, 63, 0.2) !important;
        border-radius: 0.5rem !important;
        background-color: transparent !important;
        position: relative;
        transition: border-color 0.2s ease;
    }

    /* 4. CHANGE THE TEXT INSIDE THE BUTTON */
    /* Hide the original text */
    [data-testid="stFileUploadDropzone"] button span {
        display: none !important;
    }
    /* Add our custom text */
    [data-testid="stFileUploadDropzone"] button::after {
        content: "Select data folder..." !important;
        color: #31333F !important;
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        font-size: 14px !important;
        font-weight: 400 !important;
    }

    /* Hover effect */
    [data-testid="stFileUploadDropzone"] button:hover {
        border-color: #ff4b4b !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover::after {
        color: #ff4b4b !important;
    }
</style>
""", unsafe_allow_html=True)

if "units" not in st.session_state: st.session_state.units = []
if "colors" not in st.session_state: st.session_state.colors = []
if "show" not in st.session_state: st.session_state.show = {}
if "errors" not in st.session_state: st.session_state.errors = []
if "folder_name" not in st.session_state: st.session_state.folder_name = ""

col_left, col_right = st.columns([1.2, 4], gap="medium")

with col_left:
    # 1. Folder selection (Uploader disguised by CSS)
    uploaded_files = st.file_uploader(
        "Upload folder", 
        type=["xlsx", "xls", "xlsm", "csv"], 
        accept_multiple_files='directory', 
        label_visibility="collapsed"
    )

    if uploaded_files:
        units, msg, errs = load_units(uploaded_files)
        st.session_state.units = units
        st.session_state.colors = make_colors(len(units))
        st.session_state.show = {i: True for i in range(len(units))}
        st.session_state.errors = errs
        st.session_state.folder_name = msg

    if st.session_state.folder_name:
        st.caption(f"{st.session_state.folder_name}")
    else:
        st.caption("No folder selected")
        
    st.write("")

    # 2. Filters & Table
    if st.session_state.units:
        dates = sorted({u["date"] for u in st.session_state.units})
        date_choices = ["All dates"] + [d.strftime("%d-%b-%Y") for d in dates]
        st.markdown("**Batch date**")
        sel_date = st.selectbox("Batch date", date_choices, label_visibility="collapsed")

        groups = sorted({u["group"] for u in st.session_state.units})
        st.markdown("**Group**")
        sel_group = st.selectbox("Group", ["All groups"] + groups, label_visibility="collapsed")

        filtered = [i for i, u in enumerate(st.session_state.units) if 
                    (sel_date == "All dates" or u["date"].strftime("%d-%b-%Y") == sel_date) and 
                    (sel_group == "All groups" or u["group"] == sel_group)]
    else:
        filtered = []
        sel_date = "All dates"
        sel_group = "All groups"

    st.markdown("**Samples**")
    if filtered:
        table_df = pd.DataFrame({
            "Show": [st.session_state.show.get(i, True) for i in filtered],
            "Sample": [st.session_state.units[i]["sample"] for i in filtered],
            "Annotation": [st.session_state.units[i]["annotation"] for i in filtered],
        })
        edited = st.data_editor(
            table_df,
            column_config={
                "Show": st.column_config.CheckboxColumn("Show", default=True),
                "Sample": st.column_config.TextColumn("Sample", disabled=True),
                "Annotation": st.column_config.TextColumn("Annotation", disabled=True),
            },
            hide_index=True, use_container_width=True, height=min(42 + 35 * len(table_df), 350),
            key=f"editor_{sel_date}_{sel_group}"
        )
        for i, show in zip(filtered, edited["Show"].tolist()):
            st.session_state.show[i] = bool(show)
    else:
        st.data_editor(
            pd.DataFrame(columns=["Show", "Sample", "Annotation"]),
            column_config={
                "Show": st.column_config.CheckboxColumn("Show", default=False),
                "Sample": st.column_config.TextColumn("Sample", disabled=True),
                "Annotation": st.column_config.TextColumn("Annotation", disabled=True),
            },
            hide_index=True, use_container_width=True, height=350, key="empty_table"
        )
        
    # 3. Buttons (All / None / Invert)
    b1, b2, b3 = st.columns(3)
    if b1.button("All", use_container_width=True):
        if filtered:
            for i in filtered: st.session_state.show[i] = True
            st.rerun()
    if b2.button("None", use_container_width=True):
        if filtered:
            for i in filtered: st.session_state.show[i] = False
            st.rerun()
    if b3.button("Invert", use_container_width=True):
        if filtered:
            for i in filtered: st.session_state.show[i] = not st.session_state.show.get(i, True)
            st.rerun()
            
    # 4. Toggles and Layout
    legend_on = st.checkbox("Show legend", value=False)
    layout = st.selectbox(
        "Layout", 
        ["Stacked (top / bottom)", "Side by side"], 
        label_visibility="collapsed"
    )

# Right side: Plots
with col_right:
    shown = [i for i in filtered if st.session_state.show.get(i, True)]
    
    fig = build_figure(
        st.session_state.units, 
        shown, 
        st.session_state.colors, 
        layout, 
        legend_on, 
        total_filtered=len(filtered)
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.errors:
        with st.expander(f"⚠️ {len(st.session_state.errors)} files were skipped. Click to see why."):
            for err in st.session_state.errors:
                st.write(f"- {err}")